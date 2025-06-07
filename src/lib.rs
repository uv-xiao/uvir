pub mod string_interner;
pub mod types;
pub mod ops;
pub mod region;
pub mod attribute;
pub mod error;
pub mod parser;
pub mod printer;
pub mod context;
pub mod dialects;
pub mod pass;

// Re-export macros
pub use uvir_macros::{Op, DialectType};

// Re-export dependencies used in macros
pub use inventory;
pub use smallvec;

// Re-export commonly used items
pub use context::Context;
pub use error::{Error, Result};
pub use string_interner::{StringId, StringInterner};
pub use types::{TypeId, TypeKind, FloatPrecision, DialectType as DialectTypeTrait};
pub use ops::{Op as OpTrait, OpRef, Val, Opr, Value, OpInfo, OpData, OpStorage};
pub use region::{Region, RegionId, RegionManager};
pub use attribute::{Attribute, AttributeMap, AttributeMapExt, DialectAttribute};
pub use parser::Parser;
pub use printer::Printer;
pub use pass::{RewritePattern, PatternRewriter, Pass, PassManager, PassResult, apply_patterns_greedy};