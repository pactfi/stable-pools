from pyteal import (
    Bytes,
    Expr,
    Gtxn,
    If,
    InnerTxnBuilder,
    Int,
    OnComplete,
    Seq,
    Subroutine,
    TealType,
    Txn,
    TxnType,
)
from pytealext import (
    InnerAssetTransferTxn,
    InnerPaymentTxn,
    MakeInnerApplicationCallTxn,
    MakeInnerAssetTransferTxn,
    MakeInnerPaymentTxn,
)
from pytealext.inner_transactions import InnerTxn

# Approval Program for the dummy app that will be used to boost opcode budget
# The TEAL code used to construct this program is:
# #pragma version 7
# pushint 1
DUMMY_APPROVAL_PROGRAM = "B4EB"


@Subroutine(TealType.none)
def increase_opcode_quota() -> Expr:
    """
    Increases the opcode quota of the currently running app call by roughly 690.
    """

    return MakeInnerApplicationCallTxn(
        approval_program=Bytes("base64", DUMMY_APPROVAL_PROGRAM),
        # Clear state doesn't matter, so we'll just use the same program
        clear_state_program=Bytes("base64", DUMMY_APPROVAL_PROGRAM),
        on_completion=OnComplete.DeleteApplication,
        fee=Int(0),
    )


@Subroutine(TealType.none)
def SendToCaller(asset_id: Expr, amount: Expr) -> Expr:
    """
    Send {amount} of {asset_id} to Txn.sender().

    If {asset_id} is 0, then send {amount} microAlgos to Txn.sender() instead.

    The fees are set to 0 to prevent the SSC from burning throug it's Algos.
    Therefore they must be pooled
    """
    return If(
        asset_id,  # check if it's an asset transfer (asset_id > 0)
        MakeInnerAssetTransferTxn(  # transfer the asset from SSC controlled address to caller
            asset_receiver=Txn.sender(), asset_amount=amount, xfer_asset=asset_id, fee=Int(0)
        ),
        MakeInnerPaymentTxn(  # transfer algos from SSC controlled address to the caller
            receiver=Txn.sender(), amount=amount, fee=Int(0)
        ),
    )


@Subroutine(TealType.none)
def SendToAddress(address: Expr, asset_id: Expr, amount: Expr) -> Expr:
    """
    TODO: remove (superseded by MakeInnerTransferTxn)

    Generalized version of SendToCaller, where the address can be set explicitly.

    Send {amount} of {asset_id} to {address}.

    If {asset_id} is 0, then send {amount} microAlgos to {address} instead.

    The fees are set to 0 to prevent the SSC from burning through its Algos.
    Therefore they must be pooled
    """
    return If(
        asset_id,  # check if it's an asset transfer (asset_id > 0)
        MakeInnerAssetTransferTxn(  # transfer the asset from SSC controlled address to the address
            asset_receiver=address, asset_amount=amount, xfer_asset=asset_id, fee=Int(0)
        ),
        MakeInnerPaymentTxn(  # transfer algos from SSC controlled address to the address
            receiver=address, amount=amount, fee=Int(0)
        ),
    )

