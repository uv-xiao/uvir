use crate::ops::Val;
use crate::attribute::Attribute;
use crate::region::RegionId;
// Re-export for the derive macro to use proper paths
use crate as uvir;
use crate::Op;

// Function definition operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "func", name = "func")]
pub struct FuncOp {
    #[_region]
    pub body: RegionId,
    #[_attr]
    pub sym_name: Attribute,       // Function name as symbol
    #[_attr]
    pub function_type: Attribute,  // Function signature type
    #[_attr]
    pub sym_visibility: Attribute, // Visibility (public/private)
}

// Return operation - terminates a function
#[derive(Op, Clone, Debug)]
#[operation(dialect = "func", name = "return")]
pub struct ReturnOp {
    #[_use]
    pub operands: Val,  // Return values (can be multiple)
}

// Direct function call
#[derive(Op, Clone, Debug)]
#[operation(dialect = "func", name = "call")]
pub struct CallOp {
    #[_use]
    pub operands: Val,  // Call arguments (can be multiple)
    #[_def]
    pub results: Val,   // Call results (can be multiple)
    #[_attr]
    pub callee: Attribute,  // Symbol reference to function
}

// Indirect function call
#[derive(Op, Clone, Debug)]
#[operation(dialect = "func", name = "call_indirect")]
pub struct CallIndirectOp {
    #[_use]
    pub callee: Val,    // Function pointer
    #[_use]
    pub operands: Val,  // Call arguments (can be multiple)
    #[_def]
    pub results: Val,   // Call results (can be multiple)
}

// Function constant - get a reference to a function
#[derive(Op, Clone, Debug)]
#[operation(dialect = "func", name = "constant")]
pub struct ConstantOp {
    #[_def]
    pub result: Val,    // Function reference
    #[_attr]
    pub value: Attribute,  // Symbol reference to function
}