from algosdk.transaction import StateSchema
from pyteal import (
    Addr,
    And,
    Approve,
    Assert,
    AssetHolding,
    AssetParam,
    Balance,
    Break,
    Btoi,
    Bytes,
    BytesAdd,
    BytesDiv,
    BytesEq,
    BytesGt,
    BytesLe,
    BytesMinus,
    BytesMul,
    BytesSqrt,
    Concat,
    Cond,
    Err,
    Expr,
    Global,
    Gtxn,
    If,
    InnerTxn,
    InnerTxnBuilder,
    Int,
    Itob,
    Len,
    MinBalance,
    Not,
    OnComplete,
    Pop,
    Return,
    ScratchVar,
    Seq,
    Subroutine,
    TealType,
    Txn,
    TxnField,
    TxnObject,
    TxnType,
    While,
)
from pytealext import (
    GlobalState,
    MakeInnerAssetTransferTxn,
    MulDiv64,
    SaturatingSub,
    SerializeIntegers,
)

from .contract import Contract
from .helpers.transaction import SendToAddress, SendToCaller, increase_opcode_quota
from .helpers.validation import validate_algos_transfer, validate_asset_transfer

ALGOS_UNIT_NAME = "ALGO"
UNNAMED_ASSET_PLACEHOLDER = "?"
LIQUIDITY_TOKEN_NAME_SEPARATOR = "/"
LIQUIDITY_TOKEN_NAME_SUFFIX = (
    " [SI] PACT LP TKN"  # the space is necessary as it's not added in contract code
)
LIQUIDITY_TOKEN_UNIT = "SIPLP"
LIQUIDITY_TOKEN_URL = "https://pact.fi/"
LIQUIDITY_DECIMALS = 6
LIQUIDITY_LOCK_AMOUNT = 1000
UINT_MAX = 2**64 - 1

CONTRACT_NAME = "[SI] PACT AMM"
VERSION = 200

MIN_RAMP_TIME = 86400
A_PRECISION = 10**3
MIN_A = A_PRECISION
MAX_A = 10**3 * A_PRECISION
MAX_A_CHANGE = 10
MAX_D_ITERATIONS = 64
MIN_BUDGET_THRESHOLD = 550
N_COINS = 2

# Technical - used for testing, could be used for other purposes
ITERATIONS_COUNTER_SLOT_ID = 105


@Subroutine(TealType.uint64)
def get_D(
    a_precision: Expr,
    total_primary: Expr,
    total_secondary: Expr,
    number_of_tokens: Expr,
    amp: Expr,
    nn: Expr,
) -> Expr:
    """
    Using the Newton-Raphson method of approximation, this function calculates the parameter D (invariant)
    adapted from stableswap equation:
    f(D) = (D^(n+1)/n^n * prod(x_i)) + (A*n^n - 1) * D - A*n^n*sum(x_i)
    Convergence is usually achieved within ~5 iterations.

    Args:
        precision: desired precision of convergence, default is 10^18
        total_primary: amount of primary tokens
        total_secondary: amount of secondary tokens
        number_of_tokens: number of tokens, 2 for this contract
        amp: amplifier
    """
    Dprev = ScratchVar(TealType.bytes)
    D_P = ScratchVar(TealType.bytes)
    D_P_divisor = ScratchVar(TealType.bytes)
    S = ScratchVar(TealType.bytes)
    Ann = ScratchVar(TealType.uint64)
    D = ScratchVar(TealType.bytes)
    numerator = ScratchVar(TealType.bytes)
    divisor = ScratchVar(TealType.bytes)
    i = ScratchVar(TealType.uint64, ITERATIONS_COUNTER_SLOT_ID)
    n_plus_one = ScratchVar(TealType.bytes)
    Ann_minus_precision = ScratchVar(TealType.uint64)
    return Seq(
        # in real scenarios this will not exceed 2 ^ 64 - 1
        S.store(Itob(total_primary + total_secondary)),
        If(BytesEq(S.load(), Itob(Int(0))), Return(Int(0))),
        D.store(S.load()),
        Ann.store(amp * nn),
        i.store(Int(0)),
        n_plus_one.store(Itob(number_of_tokens + Int(1))),
        Ann_minus_precision.store(Ann.load() - a_precision),
        While(i.load() < Int(MAX_D_ITERATIONS)).Do(
            Seq(
                If(Global.opcode_budget() < Int(MIN_BUDGET_THRESHOLD), increase_opcode_quota()),
                i.store(i.load() + Int(1)),
                D_P.store(BytesMul(BytesMul(D.load(), D.load()), D.load())),
                D_P_divisor.store(
                    BytesMul(
                        BytesMul(
                            Itob(total_primary),
                            Itob(total_secondary),
                        ),
                        Itob(nn),
                    )
                ),
                D_P.store(BytesDiv(D_P.load(), D_P_divisor.load())),
                Dprev.store(D.load()),
                numerator.store(
                    BytesMul(
                        D.load(),
                        BytesAdd(
                            BytesDiv(BytesMul(Itob(Ann.load()), S.load()), Itob(a_precision)),
                            BytesMul(D_P.load(), Itob(number_of_tokens)),
                        ),
                    )
                ),
                divisor.store(
                    BytesAdd(
                        BytesDiv(
                            BytesMul(Itob(Ann_minus_precision.load()), D.load()),
                            Itob(a_precision),
                        ),
                        BytesMul(n_plus_one.load(), D_P.load()),
                    )
                ),
                D.store(BytesDiv(numerator.load(), divisor.load())),
                If(BytesGt(D.load(), Dprev.load()))
                .Then(If(BytesLe(BytesMinus(D.load(), Dprev.load()), Itob(Int(1)))).Then(Break()))
                .ElseIf(BytesLe(BytesMinus(Dprev.load(), D.load()), Itob(Int(1))))
                .Then(Break()),
            )
        ),
        If(Int(MAX_D_ITERATIONS) == i.load()).Then(Err()),
        Return(Btoi(D.load())),
    )


@Subroutine(TealType.uint64)
def amplifier(initial_A: Expr, future_A: Expr, initial_A_time: Expr, future_A_time: Expr):
    """
    Calculates amplifier with gradual change over time if specified.

    Args:
        initial_A: initial value of amplifier before change
        future_A: value of amplifier after change, should be the same as initial_A if no change to amplifier applied
        initial_A_time: timestamp of starting changing the value, 0 if no change to amplifier applied
        future_A_time: timestamp of finish changing the value, 0 if no change to amplifier applied
    """
    future_initial_time_diff = future_A_time - initial_A_time
    current_initial_time_diff = Global.latest_timestamp() - initial_A_time
    return Seq(
        If(Global.latest_timestamp() < future_A_time).Then(
            If(future_A > initial_A)
            .Then(
                Return(
                    initial_A
                    + MulDiv64(
                        future_A - initial_A, current_initial_time_diff, future_initial_time_diff
                    )
                )
            )
            .Else(
                Return(
                    initial_A
                    - MulDiv64(
                        initial_A - future_A, current_initial_time_diff, future_initial_time_diff
                    )
                )
            )
        )
        # if change has ended or never happened, return final value
        .Else(Return(future_A))
    )


@Subroutine(TealType.uint64)
def get_y(other_total: Expr, amp: Expr, inv: Expr, nn: Expr, a_precision: Expr) -> Expr:
    """
    This function calculates y - in other words, how much of the swapped token should be in the pool.
    The formula is:
    f(y) = y^2 + (b-D)y - c = 0
    where:
    b = S + D/Ann
    c = D^(n+1) / n^n*P*Ann
    S = sum(x_i), i != j
    P = prod(x_i), i != j

    For pools with two tokens, S and P are the same.

    There are two solutions to this quadratic equation - one is always negative. The positive root gets returned.

    Args:
        number_of_tokens: number of tokens, 2 for this contract
        other_total: total of the non-swapped asset
        amp: amplifier
        inv: parameter D
    """
    S = other_total
    P = Itob(other_total)
    # A * n^n
    Ann = Itob(amp * nn)
    D = Itob(inv)
    a_precision_bytes = Itob(a_precision)

    b = S + Btoi(BytesDiv(BytesMul(D, a_precision_bytes), Ann))
    # c = D^3 // (4 * P * Ann)
    c = BytesDiv(
        BytesMul(BytesMul(D, BytesMul(D, D)), a_precision_bytes),
        BytesMul(Itob(Int(4)), BytesMul(P, Ann)),
    )
    b_q = ScratchVar(TealType.bytes)
    c_q = ScratchVar(TealType.bytes)
    # b_q^2 + 4 * 1 * c_q
    delta = BytesAdd(BytesMul(b_q.load(), b_q.load()), BytesMul(Itob(Int(4)), c_q.load()))
    # (-b + sqrt(delta)) // (2 * 1)
    result = ScratchVar(TealType.bytes)

    return Seq(
        If(Global.opcode_budget() < Int(MIN_BUDGET_THRESHOLD), increase_opcode_quota()),
        c_q.store(c),
        If(inv >= b)
        .Then(
            Seq(
                b_q.store(Itob(inv - b)),
                result.store(BytesDiv(BytesAdd(BytesSqrt(delta), b_q.load()), Itob(Int(2)))),
            )
        )
        .Else(
            Seq(
                b_q.store(Itob(b - inv)),
                result.store(BytesDiv(BytesMinus(BytesSqrt(delta), b_q.load()), Itob(Int(2)))),
            )
        ),
        Return(Btoi(result.load())),
    )


class StableExchangeContract(Contract):
    """
    Configurable contract allowing to exchange Algos to ASA or ASA to ASA.

    For each call to the smart contract the Txn.assets must contain [primary_asset_id, secondary_asset_id]
    """

    class STATE:
        """
        String constants representing global state vars
        """

        TOTAL_LIQUIDITY = "L"
        TOTAL_PRIMARY = "A"
        TOTAL_SECONDARY = "B"
        LIQUIDITY_TOKEN_ID = "LTID"
        CONFIG = "CONFIG"
        CONTRACT_NAME = "CONTRACT_NAME"
        VERSION = "VERSION"
        ADMIN = "ADMIN"
        INITIAL_A = "INITIAL_A"
        INITIAL_A_TIME = "INITIAL_A_TIME"
        FUTURE_A = "FUTURE_A"
        FUTURE_A_TIME = "FUTURE_A_TIME"
        FEE_BPS = "FEE_BPS"
        PACT_FEE_BPS = "PACT_FEE_BPS"
        TREASURY = "TREASURY"
        PRIMARY_FEES = "PRIMARY_FEES"
        SECONDARY_FEES = "SECONDARY_FEES"

    class ACTIONS:
        """
        String constants representing possible actions within Pact
        """

        SWAP = "SWAP"
        ADD_LIQUIDITY = "ADDLIQ"
        REMOVE_LIQUIDITY = "REMLIQ"
        WITHDRAW_FEES = "WITHDRAWFEES"
        CREATE_LIQUIDITY_TOKEN = "CLT"
        OPT_IN_TO_ASSETS = "OPTIN"  # contract opt-in to assets
        RAMP_A = "RAMP_A"
        STOP_RAMP_A = "STOP_RAMP_A"
        CHANGE_ADMIN = "CHANGE_ADMIN"
        CHANGE_PACT_FEE = "CHANGE_PACT_FEE"
        CHANGE_TREASURY = "CHANGE_TREASURY"

    def __init__(
        self,
        primary_asset_id: int,
        secondary_asset_id: int,
        fee_bps: int,
        pact_fee_bps: int,
        initial_A: int,
        admin: str,
        treasury: str,
    ):
        """
        Args:
            primary_asset_id: ID of the primary asset to be used on this exchange (must be 0 for Algos exchange)
            secondary_asset_id: ID of the secondary exchanged asset
            fee_bps: fee in basis points taken from the outcome of each swap
            pact_fee_bps: fee in basis points taken from the outcome of each swap (which goes to Pact treasury)
            initial_A: initial amplifier parameter for the stableswap invariant
            admin: address of an account that can manage contract parameters
            treasury: address of the treasury account
        """
        # The two assets must be different and in ascending order
        if primary_asset_id >= secondary_asset_id:
            raise ValueError(
                f"primary asset id ({primary_asset_id}) must be less than secondary asset id ({secondary_asset_id})"
            )
        if fee_bps >= 10000 or fee_bps < 1:
            raise ValueError("fee_bps must be in range [1, 10000)")
        if pact_fee_bps > fee_bps // 2:
            raise ValueError("pact_fee_bps cannot be greater than half of fee_bps")
        if initial_A < MIN_A or initial_A > MAX_A:
            raise ValueError(f"initial A has to be in range ({MIN_A}, {MAX_A})")

        self.primary_asset_id = Int(primary_asset_id)
        self.secondary_asset_id = Int(secondary_asset_id)
        self.tmp_fee_bps = Int(fee_bps)
        self.tmp_pact_fee_bps = Int(pact_fee_bps)
        self.number_of_tokens = Int(N_COINS)
        # n^n value need for later Ann evaluation
        self.nn = Int(N_COINS**N_COINS)
        self.a_precision = Int(A_PRECISION)
        self.tmp_admin = Addr(admin)
        self.tmp_treasury = Addr(treasury)
        self.tmp_initial_A = Int(initial_A)

        # Set up global state vars
        self.total_liquidity = GlobalState(
            StableExchangeContract.STATE.TOTAL_LIQUIDITY, TealType.uint64
        )
        self.total_primary = GlobalState(
            StableExchangeContract.STATE.TOTAL_PRIMARY, TealType.uint64
        )
        self.total_secondary = GlobalState(
            StableExchangeContract.STATE.TOTAL_SECONDARY, TealType.uint64
        )
        self.liquidity_token_id = GlobalState(
            StableExchangeContract.STATE.LIQUIDITY_TOKEN_ID, TealType.uint64
        )
        self.amm_config = GlobalState(StableExchangeContract.STATE.CONFIG, TealType.bytes)
        self.contract_name = GlobalState(StableExchangeContract.STATE.CONTRACT_NAME, TealType.bytes)
        self.version = GlobalState(StableExchangeContract.STATE.VERSION, TealType.uint64)
        self.admin = GlobalState(StableExchangeContract.STATE.ADMIN, TealType.bytes)
        # amplififer times A_PRECISION
        self.initial_A = GlobalState(StableExchangeContract.STATE.INITIAL_A, TealType.uint64)
        self.initial_A_time = GlobalState(
            StableExchangeContract.STATE.INITIAL_A_TIME, TealType.uint64
        )
        # amplififer times A_PRECISION
        self.future_A = GlobalState(StableExchangeContract.STATE.FUTURE_A, TealType.uint64)
        self.future_A_time = GlobalState(
            StableExchangeContract.STATE.FUTURE_A_TIME, TealType.uint64
        )
        self.fee_bps = GlobalState(StableExchangeContract.STATE.FEE_BPS, TealType.uint64)
        self.pact_fee_bps = GlobalState(StableExchangeContract.STATE.PACT_FEE_BPS, TealType.uint64)
        self.treasury = GlobalState(StableExchangeContract.STATE.TREASURY, TealType.bytes)
        # protocol fees used to calculate statistics in the UI
        self.protocol_fees_primary = GlobalState(
            StableExchangeContract.STATE.PRIMARY_FEES, TealType.uint64
        )
        self.protocol_fees_secondary = GlobalState(
            StableExchangeContract.STATE.SECONDARY_FEES, TealType.uint64
        )

    def get_global_schema(self) -> StateSchema:
        num_uints = 0
        num_bytes = 0
        for k, v in vars(self).items():
            if isinstance(v, GlobalState):
                if v.type_hint == TealType.uint64:
                    num_uints += 1
                elif v.type_hint == TealType.bytes:
                    num_bytes += 1
                else:
                    raise AttributeError(f"{k} (GlobalState) doesn't have specified type hint")
        return StateSchema(num_uints, num_bytes)

    def get_local_schema(self) -> StateSchema:
        return StateSchema(0, 0)

    def _validate_common_txn_fields(self) -> Expr:
        """
        Check if some common expectations about Txn are valid:
        - Txn.assets is of size at most 3 and contains the primary asset id, secondary asset id and optionally the liquidity asset id
        - Txn.rekey_to() is zero address
        These must be true with every interaction with the contract.
        """
        return And(
            Txn.rekey_to() == Global.zero_address(),
            Txn.assets[0] == self.primary_asset_id,
            Txn.assets[1] == self.secondary_asset_id,
            # don't allow more assets than expected, also if there is a third asset it must be the liquidity token ID
            Txn.assets.length() <= Int(3),
            If(Txn.assets.length() == Int(3))
            .Then(Txn.assets[2] == self.liquidity_token_id.get())
            .Else(Int(1)),  # Txn assets length < 3, no need to check index 2
        )

    def get_program(self) -> Expr:
        return Cond(
            [Txn.on_completion() == OnComplete.NoOp, self.on_noop()],
        )

    def on_noop(self) -> Expr:
        """
        Path executed when on complete is set to NoOp
        """
        return Cond(
            # Exit immediately if common checks don't pass
            [Not(self._validate_common_txn_fields()), Return(Int(0))],
            [Txn.application_id() == Int(0), self.on_create()],
            # Config
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.OPT_IN_TO_ASSETS),
                self.on_opt_in_to_assets(),
            ],
            [
                Txn.application_args[0]
                == Bytes(StableExchangeContract.ACTIONS.CREATE_LIQUIDITY_TOKEN),
                self.on_create_liquidity_token(),
            ],
            # User actions
            [Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.SWAP), self.on_swap()],
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.ADD_LIQUIDITY),
                self.on_add_liquidity(),
            ],
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.REMOVE_LIQUIDITY),
                self.on_remove_liquidity(),
            ],
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.WITHDRAW_FEES),
                self.on_withdraw_fees(),
            ],
            # Admin actions
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.RAMP_A),
                self.on_ramp_A(),
            ],
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.STOP_RAMP_A),
                self.on_stop_ramp_A(),
            ],
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.CHANGE_ADMIN),
                self.on_change_admin(),
            ],
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.CHANGE_PACT_FEE),
                self.on_change_pact_fee(),
            ],
            [
                Txn.application_args[0] == Bytes(StableExchangeContract.ACTIONS.CHANGE_TREASURY),
                self.on_change_treasury_address(),
            ],
        )

    def get_clear_program(self) -> Expr:
        # There is not much that should happen when someone clears state
        return Int(1)

    def _is_algos_exchange(self) -> Expr:
        """
        Returns:
            Expression evaluating to 1 iff the exchange is an Algos to ASA exchange (0 otherwise)
        """
        return Not(self.primary_asset_id)

    def _validate_primary_asset_deposit(self, txn: TxnObject) -> Expr:
        """
        Check if the provided transaction is a valid deposit.
        If it's an algos exchange - check Algo deposit, otherwise check asset deposit

        Returns:
            Boolean Expression checking if the deposit transfer is correct
        """
        return If(
            self._is_algos_exchange(),
            validate_algos_transfer(txn, receiver=Global.current_application_address()),
            validate_asset_transfer(
                txn, asset_id=self.primary_asset_id, receiver=Global.current_application_address()
            ),
        )

    def _validate_secondary_asset_deposit(self, txn: TxnObject) -> Expr:
        """
        Check if the provided transaction is a valid secondary asset deposit.
        """
        return validate_asset_transfer(
            txn, asset_id=self.secondary_asset_id, receiver=Global.current_application_address()
        )

    def _validate_liquidity_deposit(self, txn: TxnObject) -> Expr:
        """
        Check whether the provided txn is a valid liquidity token deposit
        """
        return validate_asset_transfer(
            txn,
            asset_id=self.liquidity_token_id.get(),
            receiver=Global.current_application_address(),
        )

    def _primary_asset_transfer_amount(self, txn: TxnObject) -> Expr:
        """
        Returns an expression evaluating to the amount of primary asset transferred in a given txn

        If Algos are exchanged on this exchange the "amount" field of the txn is returned, otherwise the "asset_amount" field is returned.
        """
        return If(self._is_algos_exchange(), txn.amount(), txn.asset_amount())

    def _secondary_asset_transfer_amount(self, txn: TxnObject) -> Expr:
        """
        Returns an expression evaluating to amount of transferred secondary asset
        """
        return txn.asset_amount()

    def _liquidity_transfer_amount(self, txn: TxnObject) -> Expr:
        """
        Returns an expression evaluating to amount of transferred liquidity asset
        """
        return txn.asset_amount()

    def on_create(self) -> Seq:
        """
        Initialize global variables
        Create liquidity token
        """
        return Seq(
            # prevent malformed contracts from being created on-chain
            # those are the same checks that we do in __init__
            Assert(self.primary_asset_id < self.secondary_asset_id),
            Assert(self.tmp_fee_bps < Int(10_000)),
            Assert(self.tmp_fee_bps),  # check that fee_bps is not 0
            Assert(self.tmp_pact_fee_bps <= (self.tmp_fee_bps / Int(2))),
            Assert(And(self.tmp_initial_A > Int(0), self.tmp_initial_A < Int(MAX_A))),
            # set the defaults for the global state variables
            self.amm_config.put(
                SerializeIntegers(
                    self.primary_asset_id,
                    self.secondary_asset_id,
                    self.tmp_fee_bps,
                    self.a_precision,
                )
            ),
            self.contract_name.put(Bytes(CONTRACT_NAME)),
            self.version.put(Int(VERSION)),
            self.admin.put(self.tmp_admin),
            self.treasury.put(self.tmp_treasury),
            self.total_primary.put(Int(0)),
            self.total_secondary.put(Int(0)),
            self.total_liquidity.put(Int(0)),
            self.liquidity_token_id.put(Int(0)),
            self.initial_A.put(self.tmp_initial_A),
            self.future_A.put(self.tmp_initial_A),
            self.initial_A_time.put(Global.latest_timestamp()),
            self.future_A_time.put(Global.latest_timestamp()),
            self.fee_bps.put(self.tmp_fee_bps),
            self.pact_fee_bps.put(self.tmp_pact_fee_bps),
            self.protocol_fees_primary.put(Int(0)),
            self.protocol_fees_secondary.put(Int(0)),
            Return(Int(1)),
        )

    def on_opt_in_to_assets(self) -> Seq:
        """
        Opt in to primary (if not algos) and secondary assets.
        Can only be executed by the creator

        Itxns: 1-2
        """
        return Seq(
            Assert(Txn.sender() == Global.creator_address()),
            If(Not(self._is_algos_exchange())).Then(
                MakeInnerAssetTransferTxn(
                    asset_receiver=Global.current_application_address(),
                    asset_amount=Int(0),
                    xfer_asset=self.primary_asset_id,
                    fee=Int(0),  # must be pooled
                )
            ),
            MakeInnerAssetTransferTxn(
                asset_receiver=Global.current_application_address(),
                asset_amount=Int(0),
                xfer_asset=self.secondary_asset_id,
                fee=Int(0),  # must be pooled
            ),
            Return(Int(1)),
        )

    def on_create_liquidity_token(self) -> Seq:
        """
        Create liquidity token for use with the exchange
        Can only be done once. Only the creator can execute this action.

        Txn.assets:
            [0]: Primary asset ID (in case it's an algos exchange, this has to be set to 0)
            [1]: Secondary asset ID

        Itxns: 1
        """
        primary_unit = ScratchVar(TealType.bytes)
        secondary_unit = ScratchVar(TealType.bytes)

        ap_primary_unit_name = AssetParam.unitName(self.primary_asset_id)
        ap_secondary_unit_name = AssetParam.unitName(self.secondary_asset_id)

        # liquidity token's name will be in format "UNIT1/UNIT2 liquidity"
        liquidity_token_name = Concat(
            primary_unit.load(),
            Bytes(LIQUIDITY_TOKEN_NAME_SEPARATOR),
            secondary_unit.load(),
            Bytes(LIQUIDITY_TOKEN_NAME_SUFFIX),
        )
        liquidity_token_unit = Bytes(LIQUIDITY_TOKEN_UNIT)
        liquidity_token_url = Bytes(LIQUIDITY_TOKEN_URL)
        return Seq(
            # Only the creator can execute this action
            Assert(Txn.sender() == Global.creator_address()),
            # make sure the liquidity token doesn't exist yet
            Assert(self.liquidity_token_id.get() == Int(0)),
            If(self._is_algos_exchange())
            .Then(
                # Primary unit name is "ALGO"
                primary_unit.store(Bytes(ALGOS_UNIT_NAME))
            )
            .Else(
                # It's an ASA to ASA exchange - load asset's unit name from the blockchain
                Seq(
                    ap_primary_unit_name,  # eval MaybeValue
                    If(ap_primary_unit_name.value() == Bytes(""))
                    .Then(
                        # Empty asset unitname, therefore set default
                        primary_unit.store(Bytes(UNNAMED_ASSET_PLACEHOLDER))
                    )
                    .Else(primary_unit.store(ap_primary_unit_name.value())),
                )
            ),
            ap_secondary_unit_name,  # eval MaybeValue
            If(ap_secondary_unit_name.value() == Bytes(""))
            .Then(
                # Secondary asset's unit name empty, so set a default
                secondary_unit.store(Bytes(UNNAMED_ASSET_PLACEHOLDER))
            )
            .Else(
                # Secondary asset's unit name found
                secondary_unit.store(ap_secondary_unit_name.value())
            ),
            # The fetched unit names are stored in slots and accesed through liquidity_token_name expr
            # Execute the asset create txn
            InnerTxnBuilder.Begin(),
            InnerTxnBuilder.SetFields(
                {
                    TxnField.fee: Int(0),  # must be pooled
                    TxnField.type_enum: TxnType.AssetConfig,
                    TxnField.config_asset_total: Int(UINT_MAX),
                    TxnField.config_asset_decimals: Int(LIQUIDITY_DECIMALS),
                    TxnField.config_asset_name: liquidity_token_name,
                    TxnField.config_asset_unit_name: liquidity_token_unit,
                    TxnField.config_asset_url: liquidity_token_url,
                    TxnField.config_asset_reserve: Global.current_application_address(),
                }
            ),
            InnerTxnBuilder.Submit(),
            self.liquidity_token_id.put(InnerTxn.created_asset_id()),
            Return(Int(1)),
        )

    def on_swap(self) -> Seq:
        """
        Swap assets for other assets, auto-detect which asset is being deposited.

        Gtxn specification:
            [i-1]: valid deposit in primary asset or secondary asset
            [i] (Txn): current application call

        Expected Txn.application_args:
            [0]: ACTION.SWAP (implied)
            [1]: expected minimum of the other token to receive from swap (prevent slippage)
        """
        deposit_txn = Gtxn[Txn.group_index() - Int(1)]
        min_expected = Btoi(Txn.application_args[1])
        return Seq(
            # check if primary asset is being deposited and the deposit is correct
            If(self._validate_primary_asset_deposit(deposit_txn))
            .Then(
                self._swap_primary(self._primary_asset_transfer_amount(deposit_txn), min_expected)
            )
            .ElseIf(self._validate_secondary_asset_deposit(deposit_txn))
            .Then(
                self._swap_secondary(
                    self._secondary_asset_transfer_amount(deposit_txn), min_expected
                )
            )
            .Else(Return(Int(0))),  # no valid deposit was made in the next transaction in a group
            Return(Int(1)),
        )

    def on_add_liquidity(self) -> Seq:
        """
        User adds liquidity to the exchange.
        Initially, if there is no liquidity in the pool they should receive a geometric mean (sqrt(a*b)) liquidity tokens.
        Otherwise the received liquidity are calculated based on difference between initial and final value of parameter D.
        The received liquidity tokens are sent back to the user.
        If the received amount of liquidity tokens is smaller than the user expects, the transaction fails.
        On imbalanced add liquidity fee is taken and added to treasury.

        Gtxn spec:
            [i-2]: Primary asset deposit
            [i-1]: Secondary asset deposit
            [i] (Txn): This application call

        Txn.application_args:
            [0]: ACTION.ADD_LIQUIDITY (implied)
            [1]: minimum expected liquidity tokens (the minimum accepted amount of liquidity tokens that the user will receive)

        Itxns: 1-2
        """
        min_expected = Btoi(Txn.application_args[1])
        primary_deposit_txn = Gtxn[Txn.group_index() - Int(2)]
        secondary_deposit_txn = Gtxn[Txn.group_index() - Int(1)]
        primary_deposit_amount = self._primary_asset_transfer_amount(primary_deposit_txn)
        secondary_deposit_amount = self._secondary_asset_transfer_amount(secondary_deposit_txn)

        amp = ScratchVar()

        initial_D = ScratchVar()
        final_D = ScratchVar()
        taxed_D = ScratchVar(TealType.uint64)

        lt_received = ScratchVar()
        lt_minted = ScratchVar()

        primary_fee = ScratchVar(TealType.uint64)
        secondary_fee = ScratchVar(TealType.uint64)

        return Seq(
            Assert(self.liquidity_token_id.get()),
            Assert(self._validate_primary_asset_deposit(primary_deposit_txn)),
            Assert(self._validate_secondary_asset_deposit(secondary_deposit_txn)),
            If(self.total_liquidity.get() == Int(0))
            .Then(
                # Initial liquidity
                Seq(
                    lt_minted.store(
                        Btoi(
                            BytesSqrt(
                                BytesMul(
                                    Itob(primary_deposit_amount), Itob(secondary_deposit_amount)
                                )
                            )
                        )
                    ),
                    # Amount is stored in global state without modifications
                    self.total_liquidity.put(lt_minted.load()),
                    self.total_primary.put(primary_deposit_amount),
                    self.total_secondary.put(secondary_deposit_amount),
                    # Lock the tokens
                    lt_minted.store(lt_minted.load() - Int(LIQUIDITY_LOCK_AMOUNT)),
                    # Check if there are enough tokens and send the liquidity tokens to the caller
                    Pop(lt_minted.load() - min_expected),
                    SendToCaller(self.liquidity_token_id.get(), lt_minted.load()),
                )
            )
            .Else(
                # There is some liquidity already present
                # Calculate the proportion of deposited amount to total tokens present in the pool
                Seq(
                    amp.store(self._amplifier()),
                    initial_D.store(
                        get_D(
                            self.a_precision,
                            self.total_primary.get(),
                            self.total_secondary.get(),
                            self.number_of_tokens,
                            amp.load(),
                            self.nn,
                        )
                    ),
                    final_D.store(
                        get_D(
                            self.a_precision,
                            self.total_primary.get() + primary_deposit_amount,
                            self.total_secondary.get() + secondary_deposit_amount,
                            self.number_of_tokens,
                            amp.load(),
                            self.nn,
                        )
                    ),
                    increase_opcode_quota(),
                    self._get_add_liq_fee(
                        old_total=self.total_primary.get(),
                        amount_added=primary_deposit_amount,
                        old_D=initial_D.load(),
                        new_D=final_D.load(),
                        fee_bps=self.fee_bps.get(),
                        result=primary_fee,
                    ),
                    self._get_add_liq_fee(
                        old_total=self.total_secondary.get(),
                        amount_added=secondary_deposit_amount,
                        old_D=initial_D.load(),
                        new_D=final_D.load(),
                        fee_bps=self.fee_bps.get(),
                        result=secondary_fee,
                    ),
                    taxed_D.store(
                        get_D(
                            self.a_precision,
                            self.total_primary.get() + primary_deposit_amount - primary_fee.load(),
                            self.total_secondary.get()
                            + secondary_deposit_amount
                            - secondary_fee.load(),
                            self.number_of_tokens,
                            amp.load(),
                            self.nn,
                        )
                    ),
                    lt_received.store(
                        MulDiv64(
                            self.total_liquidity.get(),
                            taxed_D.load() - initial_D.load(),
                            initial_D.load(),
                        )
                    ),
                    # make sure that the user receives at least {min_expected} liquidity tokens
                    # use underflow on subtraction as an error to better recognize this error on the frontend
                    Pop(lt_received.load() - min_expected),
                    # terminate the execution if no tokens were minted
                    Assert(lt_received.load() > Int(0)),
                    # Immediately send the liquidity tokens to the sender
                    SendToCaller(
                        self.liquidity_token_id.get(),
                        lt_received.load(),
                    ),
                    # Adjust the primary and secondary fees to only represent the pact fee
                    # The remainder is kept in the pool
                    primary_fee.store(
                        primary_fee.load() * self.pact_fee_bps.get() / self.fee_bps.get()
                    ),
                    secondary_fee.store(
                        secondary_fee.load() * self.pact_fee_bps.get() / self.fee_bps.get()
                    ),
                    self.total_liquidity.add_assign(lt_received.load()),
                    # update totals
                    self.total_primary.add_assign(primary_deposit_amount),
                    self.total_primary.sub_assign(primary_fee.load()),
                    self.total_secondary.add_assign(secondary_deposit_amount),
                    self.total_secondary.sub_assign(secondary_fee.load()),
                    # add fee to treasury
                    self.protocol_fees_primary.add_assign(primary_fee.load()),
                    self.protocol_fees_secondary.add_assign(secondary_fee.load()),
                ),
            ),
            Return(Int(1)),
        )

    def on_remove_liquidity(self) -> Seq:
        """
        User removes liquidity from the pool.
        This is done by depositing liquidity tokens, the contract sends back primary and secondary asset
        proportional to how many tokens are contained in the pool.
        The tokens are sent back via inner transactions.

        Gtxn spec:
            [i-1]: liquidity asset deposit
            [i] (Txn): Application call currently executing

        Txn.application_args:
            [0]: ACTION.REMOVE_LIQUIDITY (implied)
            [1]: minimum expected primary asset (the minimum accepted amount of primary asset that the user will receive)
            [2]: minimum expected secondary asset (the minimum accepted amount of secondary asset that the user will receive)
        """
        liquidity_deposit_txn = Gtxn[Txn.group_index() - Int(1)]
        min_expected_primary = Btoi(Txn.application_args[1])
        min_expected_secondary = Btoi(Txn.application_args[2])
        liquidity_deposit_amount = self._liquidity_transfer_amount(liquidity_deposit_txn)

        primary_amount = ScratchVar()  # amount of primary asset to be returned
        secondary_amount = ScratchVar()  # amount of secondary asset to be returned

        return Seq(
            Assert(self._validate_liquidity_deposit(liquidity_deposit_txn)),
            Assert(self.total_liquidity.get() != Int(0)),
            #  calculate how much of primary and secondary should be sent based on liquidity
            primary_amount.store(
                MulDiv64(
                    liquidity_deposit_amount, self.total_primary.get(), self.total_liquidity.get()
                )
            ),
            secondary_amount.store(
                MulDiv64(
                    liquidity_deposit_amount, self.total_secondary.get(), self.total_liquidity.get()
                )
            ),
            # remove liquidity and returned assets from global state
            self.total_primary.sub_assign(primary_amount.load()),
            self.total_secondary.sub_assign(secondary_amount.load()),
            self.total_liquidity.sub_assign(liquidity_deposit_amount),
            # check if both amount are bigger than minimal expected
            Pop(secondary_amount.load() - min_expected_secondary),
            Pop(primary_amount.load() - min_expected_primary),
            SendToCaller(self.primary_asset_id, primary_amount.load()),
            SendToCaller(self.secondary_asset_id, secondary_amount.load()),
            Return(Int(1)),
        )

    def _get_add_liq_fee(
        self,
        old_total: Expr,
        amount_added: Expr,
        old_D: Expr,
        new_D: Expr,
        fee_bps: Expr,
        result: ScratchVar,
    ) -> Expr:
        ideal_balance = ScratchVar(TealType.uint64)
        new_balance = ScratchVar(TealType.uint64)
        difference = ScratchVar(TealType.uint64)
        return Seq(
            ideal_balance.store(MulDiv64(new_D, old_total, old_D)),
            new_balance.store(old_total + amount_added),
            If(ideal_balance.load() > new_balance.load())
            .Then(difference.store(ideal_balance.load() - new_balance.load()))
            .Else(difference.store(new_balance.load() - ideal_balance.load())),
            result.store(
                MulDiv64(
                    difference.load(),
                    fee_bps * self.number_of_tokens,
                    Int(10_000) * (Int(4) * (self.number_of_tokens - Int(1))),
                )
            ),
        )

    def _swap_primary(self, amount: Expr, min_expected: Expr) -> Expr:
        """
        Swap `amount` of primary asset into secondary asset and send the received secondary asset to the user.
        Checks that the amount received from swap is at least `min_expected` (to prevent slippage)

        The difference in tokens is calculated via the stableswap properties described above.
        The transaction is double taxed - one to cover transaction fees, one to fund Pact's treasury.
        """
        y = ScratchVar()
        taxed_secondary_amount = ScratchVar()
        pact_fee_amt = ScratchVar()
        secondary_amount = ScratchVar()
        invariant = ScratchVar()
        return Seq(
            invariant.store(
                get_D(
                    self.a_precision,
                    self.total_primary.get(),
                    self.total_secondary.get(),
                    self.number_of_tokens,
                    self._amplifier(),
                    self.nn,
                )
            ),
            y.store(
                get_y(
                    self.total_primary.get() + amount,
                    self._amplifier(),
                    invariant.load(),
                    self.nn,
                    self.a_precision,
                )
            ),
            secondary_amount.store(self.total_secondary.get() - y.load()),
            pact_fee_amt.store(
                MulDiv64(secondary_amount.load(), self.pact_fee_bps.get(), Int(10000))
            ),
            taxed_secondary_amount.store(
                MulDiv64(
                    secondary_amount.load(),
                    (Int(10000) - self.fee_bps.get()),
                    Int(10000),
                )
            ),
            # check if slippage is not too large
            Pop(taxed_secondary_amount.load() - min_expected),
            self.total_secondary.sub_assign(pact_fee_amt.load()),
            self.total_secondary.sub_assign(taxed_secondary_amount.load()),
            self.total_primary.add_assign(amount),
            self.protocol_fees_secondary.add_assign(pact_fee_amt.load()),
            SendToCaller(self.secondary_asset_id, taxed_secondary_amount.load()),
            Return(Int(1)),
        )

    def _swap_secondary(self, amount: Expr, min_expected: Expr) -> Expr:
        """
        Swap `amount` of secondary asset into primary asset and send the received primary asset to the user.
        Checks that the amount received from swap is at least `min_expected` (to prevent slippage)

        The difference in tokens is calculated via the stableswap properties described above.
        The transaction is double taxed - one to cover transaction fees, one to fund Pact's treasury.
        """
        y = ScratchVar()
        taxed_primary_amount = ScratchVar()
        pact_fee_amt = ScratchVar()
        primary_amount = ScratchVar()
        invariant = ScratchVar()
        return Seq(
            invariant.store(
                get_D(
                    self.a_precision,
                    self.total_primary.get(),
                    self.total_secondary.get(),
                    self.number_of_tokens,
                    self._amplifier(),
                    self.nn,
                )
            ),
            y.store(
                get_y(
                    self.total_secondary.get() + amount,
                    self._amplifier(),
                    invariant.load(),
                    self.nn,
                    self.a_precision,
                )
            ),
            primary_amount.store(self.total_primary.get() - y.load()),
            pact_fee_amt.store(
                MulDiv64(primary_amount.load(), self.pact_fee_bps.get(), Int(10000))
            ),
            taxed_primary_amount.store(
                MulDiv64(
                    primary_amount.load(),
                    (Int(10000) - self.fee_bps.get()),
                    Int(10000),
                )
            ),
            # check if slippage is not too large
            Pop(taxed_primary_amount.load() - min_expected),
            self.total_primary.sub_assign(pact_fee_amt.load()),
            self.total_primary.sub_assign(taxed_primary_amount.load()),
            self.total_secondary.add_assign(amount),
            self.protocol_fees_primary.add_assign(pact_fee_amt.load()),
            SendToCaller(self.primary_asset_id, taxed_primary_amount.load()),
            Return(Int(1)),
        )

    def on_withdraw_fees(self) -> Seq:
        """
        Withdraw protocol fees to the treasury account
        """
        ah_primary = AssetHolding.balance(
            Global.current_application_address(), self.primary_asset_id
        )
        ah_secondary = AssetHolding.balance(
            Global.current_application_address(), self.secondary_asset_id
        )
        algo_balance = Balance(Global.current_application_address()) - MinBalance(
            Global.current_application_address()
        )
        primary_balance = If(
            self.primary_asset_id, Seq(ah_primary, ah_primary.value()), algo_balance
        )
        secondary_balance = Seq(ah_secondary, ah_secondary.value())

        # Saturating at 0 allows withdrawing fees in case the pool is insolvent in one currency
        surplus_primary = SaturatingSub(primary_balance, self.total_primary.get())
        surplus_secondary = SaturatingSub(secondary_balance, self.total_secondary.get())
        return Seq(
            SendToAddress(
                self.treasury.get(),
                self.primary_asset_id,
                surplus_primary,
            ),
            SendToAddress(
                self.treasury.get(),
                self.secondary_asset_id,
                surplus_secondary,
            ),
            # Withdraw surplus algos in non-algo exchange
            If(
                self.primary_asset_id,
                SendToAddress(
                    self.treasury.get(),
                    Int(0),
                    algo_balance,
                ),
            ),
            self.protocol_fees_primary.put(Int(0)),
            self.protocol_fees_secondary.put(Int(0)),
            Return(Int(1)),
        )

    # # StableExchange Admin Actions # #
    def on_ramp_A(self) -> Seq:
        """
        Sets parameters for gradual amplifier changes based on last block timestamp.
        Sets initial_A for the current amplifier value and initial_A_time for latest block timestamp.

        Txn.application_args:
            [0]: ACTION.RAMP_A (implied)
            [1]: final value of amplifier, after change multiplied by A_PRECISION
            [2]: time by which amplifier should reach final value
        """
        current_A = ScratchVar()
        future_A = Btoi(Txn.application_args[1])
        future_time = Btoi(Txn.application_args[2])
        return Seq(
            Assert(Txn.sender() == self.admin.get()),
            Assert(future_time >= Global.latest_timestamp() + Int(MIN_RAMP_TIME)),
            current_A.store(self._amplifier()),
            Assert(And(future_A >= Int(MIN_A), future_A <= Int(MAX_A))),
            If(future_A < current_A.load())
            .Then(Assert(future_A * Int(MAX_A_CHANGE) >= current_A.load()))
            .Else(Assert(future_A <= current_A.load() * Int(MAX_A_CHANGE))),
            self.initial_A.put(current_A.load()),
            self.future_A.put(future_A),
            self.initial_A_time.put(Global.latest_timestamp()),
            self.future_A_time.put(future_time),
            Approve(),
        )

    def on_stop_ramp_A(self) -> Seq:
        """
        Stops gradual change of amplifier and sets it to current value.

        Txn.application_args:
            [0]: ACTION.STOP_RAMP_A (implied)
        """
        current_A = ScratchVar()
        return Seq(
            Assert(Txn.sender() == self.admin.get()),
            current_A.store(self._amplifier()),
            self.initial_A.put(current_A.load()),
            self.future_A.put(current_A.load()),
            self.initial_A_time.put(Global.latest_timestamp()),
            self.future_A_time.put(Global.latest_timestamp()),
            Approve(),
        )

    def on_change_admin(self) -> Seq:
        """
        Sets admin address to value behind STATE.ADMIN,

        Accounts:
            [1]: valid 32-bit address of new admin

        Txn.application_args:
            [0]: ACTION.CHANGE_ADMIN (implied)
        """
        new_admin_address = Txn.accounts[1]
        return Seq(
            Assert(Txn.sender() == self.admin.get()),
            Assert(Len(new_admin_address) == Int(32)),
            self.admin.put(new_admin_address),
            Approve(),
        )

    def on_change_pact_fee(self) -> Seq:
        """
        Change the pact fee, the new pact fee has to be set through Txn.application_args.
        The new fee must be in basis points.

        Txn.application_args:
            [0]: ACTION.CHANGE_PACT_FEE (implied)
            [1]: new pact fee
        """
        new_pact_fee = Btoi(Txn.application_args[1])
        return Seq(
            Assert(Txn.sender() == self.admin.get()),
            Assert(new_pact_fee <= (self.fee_bps.get() / Int(2))),
            self.pact_fee_bps.put(new_pact_fee),
            Approve(),
        )

    def on_change_treasury_address(self) -> Seq:
        """
        Changes pact's protocol treasury address.

        Accounts:
            [1]: new, valid treasury address
        Txn.application_args:
            [0]: ACTION.CHANGE_TREASURY_ADDRESS (implied)
        """
        new_treasury_address = Txn.accounts[1]
        return Seq(
            Assert(Txn.sender() == self.admin.get()),
            Assert(Len(new_treasury_address) == Int(32)),
            self.treasury.put(new_treasury_address),
            Approve(),
        )

    def _amplifier(self):
        """
        Returns current amplifier multiplied by `self.a_precision`.
        """
        return amplifier(
            self.initial_A.get(),
            self.future_A.get(),
            self.initial_A_time.get(),
            self.future_A_time.get(),
        )
