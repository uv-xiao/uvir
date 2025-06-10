pub mod attribute;
pub mod context;
pub mod dialects;
pub mod error;
pub mod lexer;
pub mod ops;
pub mod parser;
pub mod pass;
pub mod printer;
pub mod region;
pub mod string_interner;
pub mod types;
pub mod verification;

// Re-export macros
pub use uvir_macros::{DialectType, Op};

// Re-export dependencies used in macros
pub use inventory;
pub use smallvec;

// Re-export commonly used items
pub use attribute::{Attribute, AttributeMap, AttributeMapExt, DialectAttribute};
pub use context::Context;
pub use error::{Error, Result};
pub use ops::{Op as OpTrait, OpData, OpInfo, OpRef, OpStorage, Opr, Val, Value};
pub use parser::Parser;
pub use pass::{
    apply_patterns_greedy, Pass, PassManager, PassResult, PatternRewriter, RewritePattern,
};
pub use printer::Printer;
pub use region::{Region, RegionId, RegionManager};
pub use string_interner::{StringId, StringInterner};
pub use types::{DialectType as DialectTypeTrait, FloatPrecision, TypeId, TypeKind};
