use crate::ops::Val;
use crate::attribute::Attribute;
use crate::region::RegionId;
// Re-export for the derive macro to use proper paths
use crate as uvir;
use crate::Op;

// Module operation - top-level container
#[derive(Op, Clone, Debug)]
#[operation(dialect = "builtin", name = "module")]
pub struct ModuleOp {
    #[_region]
    pub body: RegionId,
    #[_attr]
    pub sym_name: Attribute,  // Optional symbol name
}

// UnrealizedConversionCast - for progressive type system conversions
#[derive(Op, Clone, Debug)]
#[operation(dialect = "builtin", name = "unrealized_conversion_cast")]
pub struct UnrealizedConversionCastOp {
    #[_use]
    pub operands: Val,  // Can be 0-N operands
    #[_def]
    pub results: Val,   // Can be 1-N results
}

// Note: func and return operations are typically in the func dialect, not builtin
// But we can add some core type-related operations

// Constant operation for builtin attributes
#[derive(Op, Clone, Debug)]
#[operation(dialect = "builtin", name = "constant")]
pub struct ConstantOp {
    #[_def]
    pub result: Val,
    #[_attr]
    pub value: Attribute,
}

// TODO: Add more builtin operations as needed
// - tensor operations
// - memref operations  
// - other core IR operations