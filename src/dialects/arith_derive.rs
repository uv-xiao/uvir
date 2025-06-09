use crate::ops::Val;
use crate::attribute::Attribute;
// Re-export for the derive macro to use proper paths
use crate as uvir;
use crate::Op;

// Constant operation - produces a constant value
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "constant")]
pub struct ConstantOp {
    #[_def]
    pub result: Val,
    #[_attr]
    pub value: Attribute,
}

// Add operation - adds two integers
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "addi", traits = "Commutative")]
pub struct AddOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Multiply operation - multiplies two integers
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "muli", traits = "Commutative")]
pub struct MulOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Subtract operation - subtracts two integers
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "subi")]
pub struct SubOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Divide operation - divides two integers (signed)
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "divsi")]
pub struct DivSIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Divide operation - divides two integers (unsigned)
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "divui")]
pub struct DivUIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Remainder operation - computes remainder (signed)
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "remsi")]
pub struct RemSIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Remainder operation - computes remainder (unsigned)
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "remui")]
pub struct RemUIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Compare operation - compares two integers
// TODO: Enable once attribute handling is improved
// #[derive(Op, Clone, Debug)]
// #[operation(dialect = "arith", name = "cmpi")]
// pub struct CmpIOp {
//     #[_use]
//     pub lhs: Val,
//     #[_use]
//     pub rhs: Val,
//     #[_def]
//     pub result: Val,
//     #[_attr]
//     pub predicate: i64, // TODO: Should be an enum
// }

// Bitwise AND operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "andi", traits = "Commutative")]
pub struct AndIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Bitwise OR operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "ori", traits = "Commutative")]
pub struct OrIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Bitwise XOR operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "xori", traits = "Commutative")]
pub struct XOrIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Shift left operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "shli")]
pub struct ShLIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Shift right operation (logical)
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "shrui")]
pub struct ShRUIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Shift right operation (arithmetic)
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "shrsi")]
pub struct ShRSIOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Float operations

// Add operation for floats
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "addf", traits = "Commutative")]
pub struct AddFOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Subtract operation for floats
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "subf")]
pub struct SubFOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Multiply operation for floats
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "mulf", traits = "Commutative")]
pub struct MulFOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Divide operation for floats
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "divf")]
pub struct DivFOp {
    #[_use]
    pub lhs: Val,
    #[_use]
    pub rhs: Val,
    #[_def]
    pub result: Val,
}

// Negate operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "negf")]
pub struct NegFOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Compare operation for floats
// TODO: Enable once attribute handling is improved
// #[derive(Op, Clone, Debug)]
// #[operation(dialect = "arith", name = "cmpf")]
// pub struct CmpFOp {
//     #[_use]
//     pub lhs: Val,
//     #[_use]
//     pub rhs: Val,
//     #[_def]
//     pub result: Val,
//     #[_attr]
//     pub predicate: i64, // TODO: Should be an enum
// }

// Cast operations

// Sign extend
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "extsi")]
pub struct ExtSIOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Zero extend
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "extui")]
pub struct ExtUIOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Truncate integer
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "trunci")]
pub struct TruncIOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Float extend
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "extf")]
pub struct ExtFOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Float truncate
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "truncf")]
pub struct TruncFOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Signed int to float
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "sitofp")]
pub struct SIToFPOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Unsigned int to float
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "uitofp")]
pub struct UIToFPOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Float to signed int
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "fptosi")]
pub struct FPToSIOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Float to unsigned int
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "fptoui")]
pub struct FPToUIOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

// Bitcast - reinterpret bits
#[derive(Op, Clone, Debug)]
#[operation(dialect = "arith", name = "bitcast")]
pub struct BitcastOp {
    #[_use]
    pub operand: Val,
    #[_def]
    pub result: Val,
}

