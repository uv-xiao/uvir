use crate::error::{Error, Result};
use crate::context::Context;
use crate::ops::{OpData, Val, OpStorage};
use crate::types::{TypeId, TypeKind, FloatPrecision};
use crate::attribute::{Attribute, AttributeMap};
use crate::region::RegionId;
use crate::lexer::{Token};
use logos::Logos;
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;

pub struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    ctx: &'a mut Context,
    // Maps SSA value names (%0, %arg0, etc.) to Val
    value_map: HashMap<String, Val>,
    // Current region being parsed
    current_region: Option<RegionId>,
}

impl<'a> Parser<'a> {
    pub fn new(input: String, ctx: &'a mut Context) -> Result<Self> {
        let mut lexer = Token::lexer(&input);
        let mut tokens = Vec::new();
        
        while let Some(token_result) = lexer.next() {
            match token_result {
                Ok(token) => tokens.push(token),
                Err(_) => return Err(Error::ParseError(format!(
                    "Lexical error at position {}", lexer.span().start
                ))),
            }
        }
        
        let global_region = ctx.global_region();
        Ok(Self {
            tokens,
            position: 0,
            ctx,
            value_map: HashMap::new(),
            current_region: Some(global_region),
        })
    }

    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    pub fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.position)?;
        self.position += 1;
        Some(token)
    }

    pub fn expect_token(&mut self, expected: Token) -> Result<()> {
        match self.advance() {
            Some(token) if std::mem::discriminant(token) == std::mem::discriminant(&expected) => Ok(()),
            Some(token) => Err(Error::ParseError(format!(
                "Expected {:?}, found {:?}",
                expected, token
            ))),
            None => Err(Error::ParseError(format!(
                "Expected {:?}, found EOF",
                expected
            ))),
        }
    }

    pub fn expect_identifier(&mut self) -> Result<String> {
        match self.advance() {
            Some(Token::BareId(name)) => Ok(name.clone()),
            Some(token) => Err(Error::ParseError(format!(
                "Expected identifier, found {:?}",
                token
            ))),
            None => Err(Error::ParseError("Expected identifier, found EOF".to_string())),
        }
    }

    pub fn expect_integer(&mut self) -> Result<i64> {
        match self.advance() {
            Some(Token::IntegerLiteral(value)) => Ok(*value),
            Some(Token::HexIntegerLiteral(value)) => Ok(*value),
            Some(token) => Err(Error::ParseError(format!(
                "Expected integer, found {:?}",
                token
            ))),
            None => Err(Error::ParseError("Expected integer, found EOF".to_string())),
        }
    }

    pub fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    // Parse a string literal
    pub fn expect_string(&mut self) -> Result<String> {
        match self.advance() {
            Some(Token::StringLiteral(value)) => Ok(value.clone()),
            Some(token) => Err(Error::ParseError(format!(
                "Expected string literal, found {:?}",
                token
            ))),
            None => Err(Error::ParseError("Expected string literal, found EOF".to_string())),
        }
    }

    // Parse a float literal
    pub fn expect_float(&mut self) -> Result<f64> {
        match self.advance() {
            Some(Token::FloatLiteral(value)) => Ok(*value),
            Some(token) => Err(Error::ParseError(format!(
                "Expected float literal, found {:?}",
                token
            ))),
            None => Err(Error::ParseError("Expected float literal, found EOF".to_string())),
        }
    }
    
    // Parse a shape like 4x4x? or ?x10
    pub fn parse_shape(&mut self) -> Result<Vec<Option<i64>>> {
        let mut dims = Vec::new();
        
        // Handle unranked tensor with *x prefix
        if matches!(self.peek(), Some(Token::Star)) {
            self.advance();
            if matches!(self.peek(), Some(Token::X)) {
                self.advance();
            }
            return Ok(vec![]); // Empty vec represents unranked
        }
        
        // Parse dimensions with DimensionPrefix tokens
        while let Some(token) = self.peek() {
            match token {
                Token::DimensionPrefix(prefix) => {
                    let prefix = prefix.clone();
                    self.advance();
                    
                    // Parse the dimension from the prefix (e.g., "4x" -> 4)
                    let dim_str = &prefix[..prefix.len()-1]; // Remove trailing 'x'
                    if dim_str == "?" {
                        dims.push(None);
                    } else if dim_str == "*" {
                        return Ok(vec![]); // Unranked
                    } else if let Ok(dim) = dim_str.parse::<i64>() {
                        dims.push(Some(dim));
                    } else {
                        return Err(Error::ParseError(format!("Invalid dimension: {}", dim_str)));
                    }
                }
                Token::Question => {
                    self.advance();
                    dims.push(None);
                    // Check if there's an 'x' following
                    if matches!(self.peek(), Some(Token::X)) {
                        self.advance();
                    } else {
                        // Last dimension without 'x'
                        break;
                    }
                }
                Token::IntegerLiteral(_) | Token::HexIntegerLiteral(_) => {
                    dims.push(Some(self.expect_integer()?));
                    // Check if there's an 'x' following
                    if matches!(self.peek(), Some(Token::X)) {
                        self.advance();
                    } else {
                        // Last dimension without 'x'
                        break;
                    }
                }
                _ => break,
            }
        }
        
        Ok(dims)
    }
    
    // Parse a static shape (no dynamic dimensions)
    pub fn parse_static_shape(&mut self) -> Result<Vec<i64>> {
        let mut dims = Vec::new();
        
        // Parse dimensions with DimensionPrefix tokens
        while let Some(token) = self.peek() {
            match token {
                Token::DimensionPrefix(prefix) => {
                    let prefix = prefix.clone();
                    self.advance();
                    
                    // Parse the dimension from the prefix (e.g., "4x" -> 4)
                    let dim_str = &prefix[..prefix.len()-1]; // Remove trailing 'x'
                    if let Ok(dim) = dim_str.parse::<i64>() {
                        dims.push(dim);
                    } else {
                        return Err(Error::ParseError(format!("Expected static dimension, got: {}", dim_str)));
                    }
                }
                Token::IntegerLiteral(_) | Token::HexIntegerLiteral(_) => {
                    dims.push(self.expect_integer()?);
                    // Check if there's an 'x' following
                    if matches!(self.peek(), Some(Token::X)) {
                        self.advance();
                    } else {
                        // Last dimension without 'x'
                        break;
                    }
                }
                _ => break,
            }
        }
        
        Ok(dims)
    }

    // Parse a value reference like %0, %arg0, %result
    pub fn parse_value_ref(&mut self) -> Result<Val> {
        let name = match self.advance() {
            Some(Token::ValueId(name)) => name.clone(),
            Some(Token::NamedValueId(name)) => name.clone(),
            Some(token) => return Err(Error::ParseError(format!(
                "Expected value reference, found {:?}",
                token
            ))),
            None => return Err(Error::ParseError("Expected value reference, found EOF".to_string())),
        };
        
        self.value_map.get(&name)
            .copied()
            .ok_or_else(|| Error::ParseError(format!("Unknown value %{}", name)))
    }

    // Parse a type
    pub fn parse_type(&mut self) -> Result<TypeId> {
        // Check for function type
        if matches!(self.peek(), Some(Token::LeftParen)) {
            return self.parse_function_type();
        }
        
        match self.advance() {
            // Built-in integer types
            Some(Token::I1) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 1, signed: false })),
            Some(Token::I8) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 8, signed: true })),
            Some(Token::I16) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 16, signed: true })),
            Some(Token::I32) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 32, signed: true })),
            Some(Token::I64) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 64, signed: true })),
            Some(Token::U8) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 8, signed: false })),
            Some(Token::U16) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 16, signed: false })),
            Some(Token::U32) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 32, signed: false })),
            Some(Token::U64) => Ok(self.ctx.intern_type(TypeKind::Integer { width: 64, signed: false })),
            
            // Built-in float types
            Some(Token::F16) => Ok(self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Half })),
            Some(Token::F32) => Ok(self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Single })),
            Some(Token::F64) => Ok(self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Double })),
            Some(Token::BF16) => Ok(self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Half })), // BF16 treated as Half for now
            
            // Index type
            Some(Token::Index) => Ok(self.ctx.intern_type(TypeKind::Index)),
            
            // None type
            Some(Token::None) => Ok(self.ctx.intern_type(TypeKind::None)),
            
            // Complex type
            Some(Token::Complex) => {
                self.expect_token(Token::LeftAngle)?;
                let element_type = self.parse_type()?;
                self.expect_token(Token::RightAngle)?;
                Ok(self.ctx.intern_type(TypeKind::Complex { element_type }))
            }
            
            // Vector type
            Some(Token::Vector) => {
                self.expect_token(Token::LeftAngle)?;
                let shape = self.parse_static_shape()?;
                let element_type = self.parse_type()?;
                self.expect_token(Token::RightAngle)?;
                Ok(self.ctx.intern_type(TypeKind::Vector { shape, element_type }))
            }
            
            // Tensor type
            Some(Token::Tensor) => {
                self.expect_token(Token::LeftAngle)?;
                let shape = self.parse_shape()?;
                let element_type = self.parse_type()?;
                self.expect_token(Token::RightAngle)?;
                Ok(self.ctx.intern_type(TypeKind::Tensor { shape, element_type }))
            }
            
            // MemRef type
            Some(Token::Memref) => {
                self.expect_token(Token::LeftAngle)?;
                let shape = self.parse_shape()?;
                let element_type = self.parse_type()?;
                let memory_space = if matches!(self.peek(), Some(Token::Comma)) {
                    self.advance();
                    Some(self.expect_integer()? as u64)
                } else {
                    None
                };
                self.expect_token(Token::RightAngle)?;
                Ok(self.ctx.intern_type(TypeKind::MemRef { shape, element_type, memory_space }))
            }
            
            // Generic integer types
            Some(Token::GenericInteger(width)) => {
                let width = *width;
                Ok(self.ctx.intern_type(TypeKind::Integer { width, signed: true }))
            }
            Some(Token::GenericUnsigned(width)) => {
                let width = *width;
                Ok(self.ctx.intern_type(TypeKind::Integer { width, signed: false }))
            }
            Some(Token::GenericSigned(width)) => {
                let width = *width;
                Ok(self.ctx.intern_type(TypeKind::Integer { width, signed: true }))
            }
            
            // Dialect types starting with !
            Some(Token::Bang) => {
                // Parse dialect type - for now return error as not implemented
                Err(Error::ParseError("Dialect types not yet implemented".to_string()))
            }
            
            Some(token) => Err(Error::ParseError(format!(
                "Expected type, found {:?}",
                token
            ))),
            None => Err(Error::ParseError("Expected type, found EOF".to_string())),
        }
    }

    // Parse a function type like (i32, i32) -> i32
    pub fn parse_function_type(&mut self) -> Result<TypeId> {
        self.expect_token(Token::LeftParen)?;
        
        let mut inputs = Vec::new();
        while !matches!(self.peek(), Some(Token::RightParen)) {
            inputs.push(self.parse_type()?);
            
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else if !matches!(self.peek(), Some(Token::RightParen)) {
                return Err(Error::ParseError("Expected ',' or ')' in function type".to_string()));
            }
        }
        
        self.expect_token(Token::RightParen)?;
        self.expect_token(Token::Arrow)?;
        
        // Parse output types
        let mut outputs = Vec::new();
        if matches!(self.peek(), Some(Token::LeftParen)) {
            // Multiple outputs
            self.advance();
            while !matches!(self.peek(), Some(Token::RightParen)) {
                outputs.push(self.parse_type()?);
                
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.advance();
                } else if !matches!(self.peek(), Some(Token::RightParen)) {
                    return Err(Error::ParseError("Expected ',' or ')' in function outputs".to_string()));
                }
            }
            self.expect_token(Token::RightParen)?;
        } else {
            // Single output
            outputs.push(self.parse_type()?);
        }
        
        Ok(self.ctx.intern_type(TypeKind::Function { inputs, outputs }))
    }

    // Parse an attribute
    pub fn parse_attribute(&mut self) -> Result<Attribute> {
        match self.peek() {
            Some(Token::IntegerLiteral(value)) => {
                let value = *value;
                self.advance();
                
                // Check if there's a type suffix
                if matches!(self.peek(), Some(Token::Colon)) {
                    self.advance();
                    let _ty = self.parse_type()?; // Parse but ignore for now
                }
                
                Ok(Attribute::Integer(value))
            }
            Some(Token::HexIntegerLiteral(value)) => {
                let value = *value;
                self.advance();
                
                // Check if there's a type suffix
                if matches!(self.peek(), Some(Token::Colon)) {
                    self.advance();
                    let _ty = self.parse_type()?; // Parse but ignore for now
                }
                
                Ok(Attribute::Integer(value))
            }
            Some(Token::FloatLiteral(value)) => {
                let value = *value;
                self.advance();
                
                // Check if there's a type suffix
                if matches!(self.peek(), Some(Token::Colon)) {
                    self.advance();
                    let _ty = self.parse_type()?; // Parse but ignore for now
                }
                
                Ok(Attribute::Float(value))
            }
            Some(Token::StringLiteral(_)) => {
                let string = self.expect_string()?;
                
                // Check if there's a type suffix
                if matches!(self.peek(), Some(Token::Colon)) {
                    self.advance();
                    let _ty = self.parse_type()?; // Parse but ignore for now
                }
                
                Ok(Attribute::String(self.ctx.intern_string(&string)))
            }
            Some(Token::True) => {
                self.advance();
                Ok(Attribute::Integer(1)) // Represent bool as integer for now
            }
            Some(Token::False) => {
                self.advance();
                Ok(Attribute::Integer(0)) // Represent bool as integer for now
            }
            Some(Token::Unit) => {
                self.advance();
                Ok(Attribute::String(self.ctx.intern_string("unit"))) // Represent unit as string for now
            }
            Some(Token::Dense) => {
                // Dense attribute: dense<[1, 2, 3]> : tensor<3xi32>
                self.advance();
                self.expect_token(Token::LeftAngle)?;
                let value = self.parse_attribute()?; // Parse the content
                self.expect_token(Token::RightAngle)?;
                self.expect_token(Token::Colon)?;
                let _ty = self.parse_type()?; // Parse but ignore the type for now
                Ok(value) // For now, just return the inner value
            }
            Some(Token::LeftBracket) => {
                // Array attribute
                self.advance();
                let mut elements = Vec::new();
                
                while !matches!(self.peek(), Some(Token::RightBracket)) {
                    elements.push(self.parse_attribute()?);
                    
                    if matches!(self.peek(), Some(Token::Comma)) {
                        self.advance();
                    } else if !matches!(self.peek(), Some(Token::RightBracket)) {
                        return Err(Error::ParseError("Expected ',' or ']' in array attribute".to_string()));
                    }
                }
                
                self.expect_token(Token::RightBracket)?;
                Ok(Attribute::Array(elements))
            }
            // Type attribute - a type used as an attribute
            Some(token) if token.is_type() => {
                let ty = self.parse_type()?;
                Ok(Attribute::Type(ty))
            }
            Some(token) => Err(Error::ParseError(format!(
                "Unsupported attribute type: {:?}",
                token
            ))),
            None => Err(Error::ParseError("Expected attribute, found EOF".to_string())),
        }
    }

    // Parse attribute dict like {attr1 = value1, attr2 = value2}
    pub fn parse_attribute_dict(&mut self) -> Result<AttributeMap> {
        self.expect_token(Token::LeftBrace)?;
        
        let mut attrs = SmallVec::new();
        
        while !matches!(self.peek(), Some(Token::RightBrace)) {
            let name = self.expect_identifier()?;
            let name_id = self.ctx.intern_string(&name);
            
            self.expect_token(Token::Equals)?;
            
            let value = self.parse_attribute()?;
            attrs.push((name_id, value));
            
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else if !matches!(self.peek(), Some(Token::RightBrace)) {
                return Err(Error::ParseError("Expected ',' or '}' in attribute dict".to_string()));
            }
        }
        
        self.expect_token(Token::RightBrace)?;
        Ok(attrs)
    }

    // Parse a region
    pub fn parse_region(&mut self) -> Result<RegionId> {
        self.expect_token(Token::LeftBrace)?;
        
        // Create a new region
        let region_id = self.ctx.create_region();
        let saved_region = self.current_region;
        self.current_region = Some(region_id);
        
        // Parse operations in the region
        while !matches!(self.peek(), Some(Token::RightBrace)) {
            if matches!(self.peek(), Some(Token::RightBrace)) {
                break;
            }
            
            self.parse_operation()?;
        }
        
        self.expect_token(Token::RightBrace)?;
        self.current_region = saved_region;
        
        Ok(region_id)
    }

    // Parse an operation
    pub fn parse_operation(&mut self) -> Result<()> {
        // Parse optional results
        let mut results = Vec::new();
        if self.peek().map_or(false, |t| t.is_value_ref()) {
            // Parse result list
            loop {
                let result_name = match self.advance() {
                    Some(Token::ValueId(name)) => name.clone(),
                    Some(Token::NamedValueId(name)) => name.clone(),
                    _ => unreachable!(), // We checked is_value_ref above
                };
                
                // Create a new value for this result
                let placeholder_type = self.ctx.builtin_types().i32(); // Placeholder type
                let val = self.ctx.create_value(None, placeholder_type);
                self.value_map.insert(result_name, val);
                results.push(val);
                
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.advance();
                } else {
                    break;
                }
            }
            
            self.expect_token(Token::Equals)?;
        }
        
        // Parse operation name - either "dialect.opname" or dialect.opname
        let (dialect_name, op_name, is_generic) = if matches!(self.peek(), Some(Token::StringLiteral(_))) {
            // Generic syntax: "dialect.opname"
            let full_name = self.expect_string()?;
            let parts: Vec<&str> = full_name.splitn(2, '.').collect();
            if parts.len() != 2 {
                return Err(Error::ParseError(format!("Invalid operation name format: {}", full_name)));
            }
            (parts[0].to_string(), parts[1].to_string(), true)
        } else {
            // Custom syntax: dialect.opname
            let dialect_name = self.expect_identifier()?;
            self.expect_token(Token::Dot)?;
            let op_name = self.expect_identifier()?;
            (dialect_name, op_name, false)
        };
        
        // Look up the operation info
        let dialect_id = self.ctx.intern_string(&dialect_name);
        let op_name_id = self.ctx.intern_string(&op_name);
        
        let op_info = self.ctx.op_registry()
            .get(dialect_id, op_name_id)
            .ok_or_else(|| Error::ParseError(format!("Unknown operation {}.{}", dialect_name, op_name)))?;
        
        // Parse operands
        let mut operands = SmallVec::new();
        
        // Generic syntax requires parentheses for operands
        let has_parens = matches!(self.peek(), Some(Token::LeftParen));
        
        if has_parens {
            self.advance(); // consume '('
            
            while !matches!(self.peek(), Some(Token::RightParen)) {
                if self.peek().map_or(false, |t| t.is_value_ref()) {
                    operands.push(self.parse_value_ref()?);
                    
                    if matches!(self.peek(), Some(Token::Comma)) {
                        self.advance();
                    } else if !matches!(self.peek(), Some(Token::RightParen)) {
                        return Err(Error::ParseError("Expected ',' or ')' in operand list".to_string()));
                    }
                } else {
                    break;
                }
            }
            
            self.expect_token(Token::RightParen)?;
        } else {
            // Custom syntax: operands without parentheses
            while self.peek().map_or(false, |t| t.is_value_ref()) {
                operands.push(self.parse_value_ref()?);
                
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        // Parse optional attributes
        let attributes = if matches!(self.peek(), Some(Token::LeftBrace)) {
            self.parse_attribute_dict()?
        } else {
            smallvec![]
        };
        
        // Parse optional regions
        let mut regions = SmallVec::new();
        while matches!(self.peek(), Some(Token::LeftBrace)) && attributes.is_empty() {
            regions.push(self.parse_region()?);
        }
        
        // Parse type signature
        // For generic syntax, type signature is mandatory
        // For custom syntax, it's optional but we'll parse it if present
        if is_generic || matches!(self.peek(), Some(Token::Colon)) {
            if matches!(self.peek(), Some(Token::Colon)) {
                self.advance();
            } else if is_generic {
                return Err(Error::ParseError("Generic operation syntax requires type signature".to_string()));
            }
            
            // Parse function type signature
            let func_type = self.parse_function_type()?;
            
            // Update result types based on the function type
            let output_types = if let Some(TypeKind::Function { outputs, .. }) = self.ctx.get_type(func_type) {
                if outputs.len() != results.len() {
                    return Err(Error::ParseError(format!(
                        "Type signature specifies {} results but operation has {}",
                        outputs.len(), results.len()
                    )));
                }
                outputs.clone()
            } else {
                Vec::new()
            };
            
            for (i, &result) in results.iter().enumerate() {
                if let Some(_region) = self.current_region {
                    if i < output_types.len() {
                        self.ctx.set_value_type(result, output_types[i]);
                    }
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
        // Check if we have an explicit module operation
        if matches!(self.peek(), Some(Token::Module)) {
            self.advance(); // consume 'module'
            
            // Parse optional symbol name
            let _module_name = if matches!(self.peek(), Some(Token::SymbolRef(_))) {
                if let Some(Token::SymbolRef(name)) = self.advance() {
                    Some(name.clone())
                } else {
                    None
                }
            } else {
                None
            };
            
            // Parse optional attributes
            let _module_attrs = if matches!(self.peek(), Some(Token::LeftBrace)) {
                self.parse_attribute_dict()?
            } else {
                smallvec![]
            };
            
            // Parse the module body (region)
            self.expect_token(Token::LeftBrace)?;
            
            // Parse operations in the module
            while !matches!(self.peek(), Some(Token::RightBrace)) {
                if self.is_at_end() {
                    return Err(Error::ParseError("Unexpected EOF in module body".to_string()));
                }
                
                // Parse type/attribute aliases or operations
                match self.peek() {
                    Some(Token::Bang) => {
                        // Type alias: !alias = type
                        self.parse_type_alias()?;
                    }
                    Some(Token::Hash) => {
                        // Attribute alias: #alias = attribute  
                        self.parse_attribute_alias()?;
                    }
                    _ => {
                        self.parse_operation()?;
                    }
                }
            }
            
            self.expect_token(Token::RightBrace)?;
        } else {
            // Implicit module - just parse operations at top level
            while !self.is_at_end() {
                // Parse type/attribute aliases or operations
                match self.peek() {
                    Some(Token::Bang) => {
                        // Type alias: !alias = type
                        self.parse_type_alias()?;
                    }
                    Some(Token::Hash) => {
                        // Attribute alias: #alias = attribute  
                        self.parse_attribute_alias()?;
                    }
                    _ => {
                        self.parse_operation()?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    // Parse a type alias: !alias = type
    fn parse_type_alias(&mut self) -> Result<()> {
        self.expect_token(Token::Bang)?;
        let _alias_name = self.expect_identifier()?;
        self.expect_token(Token::Equals)?;
        let _aliased_type = self.parse_type()?;
        // TODO: Store type alias in context
        Ok(())
    }
    
    // Parse an attribute alias: #alias = attribute
    fn parse_attribute_alias(&mut self) -> Result<()> {
        self.expect_token(Token::Hash)?;
        let _alias_name = self.expect_identifier()?;
        self.expect_token(Token::Equals)?;
        let _aliased_attr = self.parse_attribute()?;
        // TODO: Store attribute alias in context
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parsing() {
        let mut ctx = Context::new();
        
        // Register a simple operation for testing
        let _dialect_id = ctx.intern_string("test");
        let _op_name_id = ctx.intern_string("op");
        
        let input = r#"%0 = test.op : i32"#;
        let parser = Parser::new(input.to_string(), &mut ctx).unwrap();
        
        // This would fail until we properly register the operation
        // Just test that the lexer works
        assert!(!parser.is_at_end());
    }

    #[test]
    fn test_lexer_integration() {
        let mut ctx = Context::new();
        let input = r#"func.call %0, %arg1 : (i32, i32) -> i32"#;
        let parser = Parser::new(input.to_string(), &mut ctx).unwrap();
        
        // Should successfully tokenize without errors
        assert!(!parser.tokens.is_empty());
    }

    #[test]
    fn test_simple_tokenization() {
        let mut ctx = Context::new();
        let input = "i32 f64 %0 %arg1";
        let parser = Parser::new(input.to_string(), &mut ctx).unwrap();
        
        // Check that we have the expected tokens
        assert_eq!(parser.tokens.len(), 4);
        assert!(matches!(parser.tokens[0], Token::I32));
        assert!(matches!(parser.tokens[1], Token::F64));
        assert!(matches!(parser.tokens[2], Token::ValueId(_)));
        assert!(matches!(parser.tokens[3], Token::NamedValueId(_)));
    }
}