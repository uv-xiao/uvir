use crate::error::{Error, Result};
use crate::context::Context;
use crate::ops::{OpData, Val, OpStorage};
use crate::types::{TypeId, TypeKind, FloatPrecision};
use crate::attribute::{Attribute, AttributeMap};
use crate::region::RegionId;
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;

pub struct Parser<'a> {
    input: String,
    position: usize,
    ctx: &'a mut Context,
    // Maps SSA value names (%0, %arg0, etc.) to Val
    value_map: HashMap<String, Val>,
    // Current region being parsed
    current_region: Option<RegionId>,
}

impl<'a> Parser<'a> {
    pub fn new(input: String, ctx: &'a mut Context) -> Self {
        let global_region = ctx.global_region();
        Self {
            input,
            position: 0,
            ctx,
            value_map: HashMap::new(),
            current_region: Some(global_region),
        }
    }

    pub fn peek(&self) -> Option<char> {
        self.input.chars().nth(self.position)
    }

    pub fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.position += ch.len_utf8();
        Some(ch)
    }

    pub fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    pub fn expect_char(&mut self, expected: char) -> Result<()> {
        self.skip_whitespace();
        match self.advance() {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(Error::ParseError(format!(
                "Expected '{}', found '{}'",
                expected, ch
            ))),
            None => Err(Error::ParseError(format!(
                "Expected '{}', found EOF",
                expected
            ))),
        }
    }

    pub fn parse_identifier(&mut self) -> Result<String> {
        self.skip_whitespace();
        let start = self.position;
        
        if !self.peek().map_or(false, |ch| ch.is_alphabetic() || ch == '_') {
            return Err(Error::ParseError("Expected identifier".to_string()));
        }

        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }

        Ok(self.input[start..self.position].to_string())
    }

    pub fn parse_integer(&mut self) -> Result<i64> {
        self.skip_whitespace();
        let start = self.position;
        
        if self.peek() == Some('-') {
            self.advance();
        }

        if !self.peek().map_or(false, |ch| ch.is_numeric()) {
            return Err(Error::ParseError("Expected integer".to_string()));
        }

        while let Some(ch) = self.peek() {
            if ch.is_numeric() {
                self.advance();
            } else {
                break;
            }
        }

        self.input[start..self.position]
            .parse()
            .map_err(|_| Error::ParseError("Invalid integer".to_string()))
    }

    pub fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }

    // Parse a string literal
    pub fn parse_string(&mut self) -> Result<String> {
        self.skip_whitespace();
        self.expect_char('"')?;
        
        let mut result = String::new();
        let mut escaped = false;
        
        loop {
            match self.advance() {
                Some('"') if !escaped => break,
                Some('\\') if !escaped => escaped = true,
                Some(ch) => {
                    if escaped {
                        match ch {
                            'n' => result.push('\n'),
                            'r' => result.push('\r'),
                            't' => result.push('\t'),
                            '\\' => result.push('\\'),
                            '"' => result.push('"'),
                            _ => return Err(Error::ParseError(format!("Invalid escape sequence \\{}", ch))),
                        }
                        escaped = false;
                    } else {
                        result.push(ch);
                    }
                }
                None => return Err(Error::ParseError("Unterminated string".to_string())),
            }
        }
        
        Ok(result)
    }

    // Parse a float literal
    pub fn parse_float(&mut self) -> Result<f64> {
        self.skip_whitespace();
        let start = self.position;
        
        if self.peek() == Some('-') {
            self.advance();
        }

        // Parse integer part
        if !self.peek().map_or(false, |ch| ch.is_numeric()) {
            return Err(Error::ParseError("Expected float".to_string()));
        }

        while let Some(ch) = self.peek() {
            if ch.is_numeric() {
                self.advance();
            } else {
                break;
            }
        }

        // Check for decimal point
        if self.peek() == Some('.') {
            self.advance();
            
            // Parse fractional part
            while let Some(ch) = self.peek() {
                if ch.is_numeric() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Check for exponent
        if let Some('e') | Some('E') = self.peek() {
            self.advance();
            if let Some('+') | Some('-') = self.peek() {
                self.advance();
            }
            while let Some(ch) = self.peek() {
                if ch.is_numeric() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.input[start..self.position]
            .parse()
            .map_err(|_| Error::ParseError("Invalid float".to_string()))
    }

    // Parse a value reference like %0, %arg0, %result
    pub fn parse_value_ref(&mut self) -> Result<Val> {
        self.skip_whitespace();
        self.expect_char('%')?;
        
        let name = if self.peek().map_or(false, |ch| ch.is_numeric()) {
            // Parse numeric value like %0, %1
            let num = self.parse_integer()?;
            format!("{}", num)
        } else {
            // Parse named value like %arg0, %result
            self.parse_identifier()?
        };
        
        self.value_map.get(&name)
            .copied()
            .ok_or_else(|| Error::ParseError(format!("Unknown value %{}", name)))
    }

    // Parse a type
    pub fn parse_type(&mut self) -> Result<TypeId> {
        self.skip_whitespace();
        
        // Check for function type
        if self.peek() == Some('(') {
            return self.parse_function_type();
        }
        
        let type_name = self.parse_identifier()?;
        
        match type_name.as_str() {
            // Integer types
            "i1" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 1, signed: false })),
            "i8" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 8, signed: true })),
            "i16" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 16, signed: true })),
            "i32" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 32, signed: true })),
            "i64" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 64, signed: true })),
            "u8" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 8, signed: false })),
            "u16" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 16, signed: false })),
            "u32" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 32, signed: false })),
            "u64" => Ok(self.ctx.intern_type(TypeKind::Integer { width: 64, signed: false })),
            
            // Float types
            "f16" => Ok(self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Half })),
            "f32" => Ok(self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Single })),
            "f64" => Ok(self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Double })),
            
            // Generic integer type i<N>
            _ if type_name.starts_with('i') => {
                let width: u32 = type_name[1..].parse()
                    .map_err(|_| Error::ParseError(format!("Invalid integer type: {}", type_name)))?;
                Ok(self.ctx.intern_type(TypeKind::Integer { width, signed: true }))
            }
            
            // Generic unsigned type u<N>
            _ if type_name.starts_with('u') => {
                let width: u32 = type_name[1..].parse()
                    .map_err(|_| Error::ParseError(format!("Invalid unsigned type: {}", type_name)))?;
                Ok(self.ctx.intern_type(TypeKind::Integer { width, signed: false }))
            }
            
            // Dialect type
            _ => {
                // Could be a dialect type like tensor<4x4xf32>
                // For now, we'll return an error
                Err(Error::ParseError(format!("Unknown type: {}", type_name)))
            }
        }
    }

    // Parse a function type like (i32, i32) -> i32
    pub fn parse_function_type(&mut self) -> Result<TypeId> {
        self.expect_char('(')?;
        
        let mut inputs = Vec::new();
        while self.peek() != Some(')') {
            inputs.push(self.parse_type()?);
            
            self.skip_whitespace();
            if self.peek() == Some(',') {
                self.advance();
            } else if self.peek() != Some(')') {
                return Err(Error::ParseError("Expected ',' or ')' in function type".to_string()));
            }
        }
        
        self.expect_char(')')?;
        self.skip_whitespace();
        
        // Expect ->
        if self.advance() != Some('-') || self.advance() != Some('>') {
            return Err(Error::ParseError("Expected '->' in function type".to_string()));
        }
        
        self.skip_whitespace();
        
        // Parse output types
        let mut outputs = Vec::new();
        if self.peek() == Some('(') {
            // Multiple outputs
            self.advance();
            while self.peek() != Some(')') {
                outputs.push(self.parse_type()?);
                
                self.skip_whitespace();
                if self.peek() == Some(',') {
                    self.advance();
                } else if self.peek() != Some(')') {
                    return Err(Error::ParseError("Expected ',' or ')' in function outputs".to_string()));
                }
            }
            self.expect_char(')')?;
        } else {
            // Single output
            outputs.push(self.parse_type()?);
        }
        
        Ok(self.ctx.intern_type(TypeKind::Function { inputs, outputs }))
    }

    // Parse an attribute
    pub fn parse_attribute(&mut self) -> Result<Attribute> {
        self.skip_whitespace();
        
        // Check for integer
        if self.peek().map_or(false, |ch| ch.is_numeric() || ch == '-') {
            // Could be integer or float
            let _start = self.position;
            let mut is_float = false;
            
            // Look ahead to determine if it's a float
            let saved_pos = self.position;
            if self.peek() == Some('-') {
                self.advance();
            }
            while let Some(ch) = self.peek() {
                if ch == '.' || ch == 'e' || ch == 'E' {
                    is_float = true;
                    break;
                }
                if !ch.is_numeric() {
                    break;
                }
                self.advance();
            }
            self.position = saved_pos;
            
            if is_float {
                Ok(Attribute::Float(self.parse_float()?))
            } else {
                Ok(Attribute::Integer(self.parse_integer()?))
            }
        } else if self.peek() == Some('"') {
            // String attribute
            let string = self.parse_string()?;
            Ok(Attribute::String(self.ctx.intern_string(&string)))
        } else if self.peek() == Some('[') {
            // Array attribute
            self.advance();
            let mut elements = Vec::new();
            
            while self.peek() != Some(']') {
                elements.push(self.parse_attribute()?);
                
                self.skip_whitespace();
                if self.peek() == Some(',') {
                    self.advance();
                } else if self.peek() != Some(']') {
                    return Err(Error::ParseError("Expected ',' or ']' in array attribute".to_string()));
                }
            }
            
            self.expect_char(']')?;
            Ok(Attribute::Array(elements))
        } else {
            // Could be a type attribute or other
            Err(Error::ParseError("Unsupported attribute type".to_string()))
        }
    }

    // Parse attribute dict like {attr1 = value1, attr2 = value2}
    pub fn parse_attribute_dict(&mut self) -> Result<AttributeMap> {
        self.skip_whitespace();
        self.expect_char('{')?;
        
        let mut attrs = SmallVec::new();
        
        while self.peek() != Some('}') {
            self.skip_whitespace();
            let name = self.parse_identifier()?;
            let name_id = self.ctx.intern_string(&name);
            
            self.skip_whitespace();
            self.expect_char('=')?;
            
            let value = self.parse_attribute()?;
            attrs.push((name_id, value));
            
            self.skip_whitespace();
            if self.peek() == Some(',') {
                self.advance();
            } else if self.peek() != Some('}') {
                return Err(Error::ParseError("Expected ',' or '}' in attribute dict".to_string()));
            }
        }
        
        self.expect_char('}')?;
        Ok(attrs)
    }

    // Parse a region
    pub fn parse_region(&mut self) -> Result<RegionId> {
        self.skip_whitespace();
        self.expect_char('{')?;
        
        // Create a new region
        let region_id = self.ctx.create_region();
        let saved_region = self.current_region;
        self.current_region = Some(region_id);
        
        // Parse operations in the region
        while self.peek() != Some('}') {
            self.skip_whitespace();
            if self.peek() == Some('}') {
                break;
            }
            
            self.parse_operation()?;
        }
        
        self.expect_char('}')?;
        self.current_region = saved_region;
        
        Ok(region_id)
    }

    // Parse an operation
    pub fn parse_operation(&mut self) -> Result<()> {
        self.skip_whitespace();
        
        // Parse optional results
        let mut results = Vec::new();
        if self.peek() == Some('%') {
            // Parse result list
            loop {
                self.expect_char('%')?;
                let result_name = if self.peek().map_or(false, |ch| ch.is_numeric()) {
                    let num = self.parse_integer()?;
                    format!("{}", num)
                } else {
                    self.parse_identifier()?
                };
                
                // Create a new value for this result
                let placeholder_type = self.ctx.builtin_types().i32(); // Placeholder type
                let val = self.ctx.create_value(None, placeholder_type);
                self.value_map.insert(result_name, val);
                results.push(val);
                
                self.skip_whitespace();
                if self.peek() == Some(',') {
                    self.advance();
                    self.skip_whitespace();
                } else {
                    break;
                }
            }
            
            self.skip_whitespace();
            self.expect_char('=')?;
            self.skip_whitespace();
        }
        
        // Parse operation name (dialect.opname)
        let dialect_name = self.parse_identifier()?;
        self.expect_char('.')?;
        let op_name = self.parse_identifier()?;
        
        // Look up the operation info
        let dialect_id = self.ctx.intern_string(&dialect_name);
        let op_name_id = self.ctx.intern_string(&op_name);
        
        let op_info = self.ctx.op_registry()
            .get(dialect_id, op_name_id)
            .ok_or_else(|| Error::ParseError(format!("Unknown operation {}.{}", dialect_name, op_name)))?;
        
        // Parse operands
        self.skip_whitespace();
        let mut operands = SmallVec::new();
        
        // Check if there are operands (not attributes or regions)
        if self.peek() == Some('%') {
            loop {
                operands.push(self.parse_value_ref()?);
                
                self.skip_whitespace();
                if self.peek() == Some(',') {
                    self.advance();
                    self.skip_whitespace();
                } else {
                    break;
                }
            }
        }
        
        // Parse optional attributes
        let attributes = if self.peek() == Some('{') {
            self.parse_attribute_dict()?
        } else {
            smallvec![]
        };
        
        // Parse optional regions
        let mut regions = SmallVec::new();
        while self.peek() == Some('{') && attributes.is_empty() {
            regions.push(self.parse_region()?);
            self.skip_whitespace();
        }
        
        // Parse result types
        if !results.is_empty() {
            self.skip_whitespace();
            if self.peek() == Some(':') {
                self.advance();
                self.skip_whitespace();
                
                // Parse type list
                if results.len() == 1 {
                    let ty = self.parse_type()?;
                    // Update the result value with the correct type
                    if let Some(_region) = self.current_region {
                        self.ctx.set_value_type(results[0], ty);
                    }
                } else {
                    self.expect_char('(')?;
                    for (i, &result) in results.iter().enumerate() {
                        if i > 0 {
                            self.expect_char(',')?;
                            self.skip_whitespace();
                        }
                        let ty = self.parse_type()?;
                        if let Some(_region) = self.current_region {
                            self.ctx.set_value_type(result, ty);
                        }
                    }
                    self.expect_char(')')?;
                }
            }
        }
        
        // Create the operation
        let op_data = OpData {
            info: op_info,
            operands,
            results: results.into(),
            attributes,
            regions,
            custom_data: OpStorage::new(), // TODO: Parse custom data if needed
        };
        
        // Add operation to current region
        if let Some(region) = self.current_region {
            self.ctx.add_operation(region, op_data)?;
        }
        
        Ok(())
    }

    // Parse a module (top-level)
    pub fn parse_module(&mut self) -> Result<()> {
        self.skip_whitespace();
        
        while !self.is_at_end() {
            self.skip_whitespace();
            if self.is_at_end() {
                break;
            }
            
            self.parse_operation()?;
            self.skip_whitespace();
        }
        
        Ok(())
    }
}