use std::fmt::Write;
use crate::error::Result;
use crate::context::Context;
use crate::ops::{OpData, Val};
use crate::types::{TypeId, TypeKind, FloatPrecision};
use crate::attribute::Attribute;
use crate::region::RegionId;

pub struct Printer {
    output: String,
    indent_level: usize,
    indent_str: String,
}

impl Printer {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent_level: 0,
            indent_str: "  ".to_string(),
        }
    }

    pub fn print(&mut self, s: &str) -> Result<()> {
        self.output.push_str(s);
        Ok(())
    }

    pub fn println(&mut self, s: &str) -> Result<()> {
        writeln!(&mut self.output, "{}", s)
            .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
    }

    pub fn print_indent(&mut self) -> Result<()> {
        for _ in 0..self.indent_level {
            self.output.push_str(&self.indent_str);
        }
        Ok(())
    }

    pub fn indent(&mut self) {
        self.indent_level += 1;
    }

    pub fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    pub fn get_output(self) -> String {
        self.output
    }

    pub fn clear(&mut self) {
        self.output.clear();
        self.indent_level = 0;
    }

    // Print a value reference like %0, %arg0
    pub fn print_value(&mut self, ctx: &Context, val: Val) -> Result<()> {
        if let Some(region) = ctx.get_region(ctx.global_region()) {
            if let Some(value) = region.get_value(val) {
                if let Some(name) = value.name {
                    if let Some(name_str) = ctx.get_string(name) {
                        write!(&mut self.output, "%{}", name_str)
                            .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
                    } else {
                        write!(&mut self.output, "%{:?}", val)
                            .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
                    }
                } else {
                    // Use a numeric name based on the slot map key
                    write!(&mut self.output, "%{:?}", val)
                        .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
                }
            } else {
                write!(&mut self.output, "%{:?}", val)
                    .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
            }
        } else {
            write!(&mut self.output, "%{:?}", val)
                .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
        }
    }

    // Print a type
    pub fn print_type(&mut self, ctx: &Context, ty: TypeId) -> Result<()> {
        if let Some(type_kind) = ctx.get_type(ty) {
            match type_kind {
                TypeKind::Integer { width, signed } => {
                    if *signed || *width == 1 {
                        write!(&mut self.output, "i{}", width)
                    } else {
                        write!(&mut self.output, "u{}", width)
                    }
                    .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
                }
                TypeKind::Float { precision } => {
                    let s = match precision {
                        FloatPrecision::Half => "f16",
                        FloatPrecision::Single => "f32",
                        FloatPrecision::Double => "f64",
                    };
                    self.print(s)
                }
                TypeKind::Function { inputs, outputs } => {
                    self.print("(")?;
                    for (i, &input) in inputs.iter().enumerate() {
                        if i > 0 {
                            self.print(", ")?;
                        }
                        self.print_type(ctx, input)?;
                    }
                    self.print(") -> ")?;
                    
                    if outputs.len() == 1 {
                        self.print_type(ctx, outputs[0])
                    } else {
                        self.print("(")?;
                        for (i, &output) in outputs.iter().enumerate() {
                            if i > 0 {
                                self.print(", ")?;
                            }
                            self.print_type(ctx, output)?;
                        }
                        self.print(")")
                    }
                }
                TypeKind::Dialect { dialect, data: _ } => {
                    // For now, just print the dialect name
                    if let Some(dialect_name) = ctx.get_string(*dialect) {
                        write!(&mut self.output, "!{}.type", dialect_name)
                            .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
                    } else {
                        Err(crate::error::Error::InternalError("Unknown dialect".to_string()))
                    }
                }
            }
        } else {
            Err(crate::error::Error::InternalError("Unknown type".to_string()))
        }
    }

    // Print an attribute
    pub fn print_attribute(&mut self, ctx: &Context, attr: &Attribute) -> Result<()> {
        match attr {
            Attribute::Integer(i) => {
                write!(&mut self.output, "{}", i)
                    .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
            }
            Attribute::Float(f) => {
                write!(&mut self.output, "{}", f)
                    .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
            }
            Attribute::String(s) => {
                if let Some(string) = ctx.get_string(*s) {
                    write!(&mut self.output, "\"{}\"", string.escape_default())
                        .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
                } else {
                    Err(crate::error::Error::InternalError("Unknown string".to_string()))
                }
            }
            Attribute::Type(ty) => {
                self.print_type(ctx, *ty)
            }
            Attribute::Array(elements) => {
                self.print("[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        self.print(", ")?;
                    }
                    self.print_attribute(ctx, elem)?;
                }
                self.print("]")
            }
            Attribute::Dialect { dialect, data: _ } => {
                // For now, just print the dialect name
                if let Some(dialect_name) = ctx.get_string(*dialect) {
                    write!(&mut self.output, "#{}.<attr>", dialect_name)
                        .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
                } else {
                    Err(crate::error::Error::InternalError("Unknown dialect".to_string()))
                }
            }
        }
    }

    // Print an operation
    pub fn print_operation(&mut self, ctx: &Context, op: &OpData) -> Result<()> {
        // Print results if any
        if !op.results.is_empty() {
            for (i, &result) in op.results.iter().enumerate() {
                if i > 0 {
                    self.print(", ")?;
                }
                self.print_value(ctx, result)?;
            }
            self.print(" = ")?;
        }

        // Print operation name
        self.print(&format!("{}.{}", op.info.dialect, op.info.name))?;

        // Print operands
        if !op.operands.is_empty() {
            self.print(" ")?;
            for (i, &operand) in op.operands.iter().enumerate() {
                if i > 0 {
                    self.print(", ")?;
                }
                self.print_value(ctx, operand)?;
            }
        }

        // Print attributes
        if !op.attributes.is_empty() {
            self.print(" {")?;
            for (i, (key, value)) in op.attributes.iter().enumerate() {
                if i > 0 {
                    self.print(", ")?;
                }
                if let Some(key_name) = ctx.get_string(*key) {
                    self.print(key_name)?;
                    self.print(" = ")?;
                    self.print_attribute(ctx, value)?;
                }
            }
            self.print("}")?;
        }

        // Print regions
        for region_id in &op.regions {
            self.print(" ")?;
            self.print_region(ctx, *region_id)?;
        }

        // Print function type signature (operand types -> result types)
        self.print(" : ")?;
        
        // Print operand types
        if op.operands.is_empty() {
            self.print("()")?;
        } else if op.operands.len() == 1 {
            if let Some(region) = ctx.get_region(ctx.global_region()) {
                if let Some(value) = region.get_value(op.operands[0]) {
                    self.print_type(ctx, value.ty)?;
                }
            }
        } else {
            self.print("(")?;
            for (i, &operand) in op.operands.iter().enumerate() {
                if i > 0 {
                    self.print(", ")?;
                }
                if let Some(region) = ctx.get_region(ctx.global_region()) {
                    if let Some(value) = region.get_value(operand) {
                        self.print_type(ctx, value.ty)?;
                    }
                }
            }
            self.print(")")?;
        }
        
        self.print(" -> ")?;
        
        // Print result types
        if op.results.is_empty() {
            self.print("()")?;
        } else if op.results.len() == 1 {
            if let Some(region) = ctx.get_region(ctx.global_region()) {
                if let Some(value) = region.get_value(op.results[0]) {
                    self.print_type(ctx, value.ty)?;
                }
            }
        } else {
            self.print("(")?;
            for (i, &result) in op.results.iter().enumerate() {
                if i > 0 {
                    self.print(", ")?;
                }
                if let Some(region) = ctx.get_region(ctx.global_region()) {
                    if let Some(value) = region.get_value(result) {
                        self.print_type(ctx, value.ty)?;
                    }
                }
            }
            self.print(")")?;
        }

        Ok(())
    }

    // Print a region
    pub fn print_region(&mut self, ctx: &Context, region_id: RegionId) -> Result<()> {
        self.println("{")?;
        self.indent();

        if let Some(region) = ctx.get_region(region_id) {
            for (_opr, op) in region.iter_ops() {
                self.print_indent()?;
                self.print_operation(ctx, op)?;
                self.println("")?;
            }
        }

        self.dedent();
        self.print_indent()?;
        self.print("}")
    }

    // Print a module (top-level)
    pub fn print_module(&mut self, ctx: &Context) -> Result<()> {
        let global_region = ctx.global_region();
        if let Some(region) = ctx.get_region(global_region) {
            for (_opr, op) in region.iter_ops() {
                self.print_operation(ctx, op)?;
                self.println("")?;
            }
        }
        Ok(())
    }
}

impl Default for Printer {
    fn default() -> Self {
        Self::new()
    }
}