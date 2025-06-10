use crate::error::Result;
use crate::impl_dialect_type;
use crate::parser::Parser;
use crate::printer::Printer;
use crate::types::{FloatPrecision, TypeId, TypeKind};

#[derive(Clone, PartialEq, Debug)]
pub struct IntegerType {
    pub width: u32,
    pub signed: bool,
}

impl IntegerType {
    pub fn parse(_parser: &mut Parser) -> Result<Self> {
        // TODO: Implement dialect-specific type parsing with new token-based parser
        // For now, built-in types are handled directly by the lexer
        Err(crate::error::Error::ParseError(
            "Not implemented yet".to_string(),
        ))
    }

    pub fn print(&self, printer: &mut Printer) -> Result<()> {
        printer.print(&format!("i{}", self.width))
    }
}

impl_dialect_type!(IntegerType);

#[derive(Clone, PartialEq, Debug)]
pub struct UnsignedType {
    pub width: u32,
}

impl UnsignedType {
    pub fn parse(_parser: &mut Parser) -> Result<Self> {
        // TODO: Implement dialect-specific type parsing with new token-based parser
        // For now, built-in types are handled directly by the lexer
        Err(crate::error::Error::ParseError(
            "Not implemented yet".to_string(),
        ))
    }

    pub fn print(&self, printer: &mut Printer) -> Result<()> {
        printer.print(&format!("u{}", self.width))
    }
}

impl_dialect_type!(UnsignedType);

#[derive(Clone, PartialEq, Debug)]
pub struct FloatType {
    pub precision: FloatPrecision,
}

impl FloatType {
    pub fn parse(_parser: &mut Parser) -> Result<Self> {
        // TODO: Implement dialect-specific type parsing with new token-based parser
        // For now, built-in types are handled directly by the lexer
        Err(crate::error::Error::ParseError(
            "Not implemented yet".to_string(),
        ))
    }

    pub fn print(&self, printer: &mut Printer) -> Result<()> {
        let width = match self.precision {
            FloatPrecision::Half => 16,
            FloatPrecision::Single => 32,
            FloatPrecision::Double => 64,
        };
        printer.print(&format!("f{}", width))
    }
}

impl_dialect_type!(FloatType);

pub fn integer_type(ctx: &mut crate::Context, width: u32, signed: bool) -> TypeId {
    ctx.intern_type(TypeKind::Integer { width, signed })
}

pub fn float_type(ctx: &mut crate::Context, precision: FloatPrecision) -> TypeId {
    ctx.intern_type(TypeKind::Float { precision })
}

pub fn function_type(
    ctx: &mut crate::Context,
    inputs: Vec<TypeId>,
    outputs: Vec<TypeId>,
) -> TypeId {
    ctx.intern_type(TypeKind::Function { inputs, outputs })
}
