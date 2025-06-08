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
            
            // Generic integer types
            Some(Token::GenericInteger(width)) => {
                let width = *width;
                Ok(self.ctx.intern_type(TypeKind::Integer { width, signed: true }))
            }
            Some(Token::GenericUnsigned(width)) => {
                let width = *width;
                Ok(self.ctx.intern_type(TypeKind::Integer { width, signed: false }))
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
                Ok(Attribute::Integer(value))
            }
            Some(Token::HexIntegerLiteral(value)) => {
                let value = *value;
                self.advance();
                Ok(Attribute::Integer(value))
            }
            Some(Token::FloatLiteral(value)) => {
                let value = *value;
                self.advance();
                Ok(Attribute::Float(value))
            }
            Some(Token::StringLiteral(_)) => {
                let string = self.expect_string()?;
                Ok(Attribute::String(self.ctx.intern_string(&string)))
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
        
        // Parse operation name (dialect.opname)
        let dialect_name = self.expect_identifier()?;
        self.expect_token(Token::Dot)?;
        let op_name = self.expect_identifier()?;
        
        // Look up the operation info
        let dialect_id = self.ctx.intern_string(&dialect_name);
        let op_name_id = self.ctx.intern_string(&op_name);
        
        let op_info = self.ctx.op_registry()
            .get(dialect_id, op_name_id)
            .ok_or_else(|| Error::ParseError(format!("Unknown operation {}.{}", dialect_name, op_name)))?;
        
        // Parse operands
        let mut operands = SmallVec::new();
        
        // Check if there are operands (value references)
        while self.peek().map_or(false, |t| t.is_value_ref()) {
            operands.push(self.parse_value_ref()?);
            
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else {
                break;
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
        
        // Parse result types
        if !results.is_empty() {
            if matches!(self.peek(), Some(Token::Colon)) {
                self.advance();
                
                // Parse type list
                if results.len() == 1 {
                    let ty = self.parse_type()?;
                    // Update the result value with the correct type
                    if let Some(_region) = self.current_region {
                        self.ctx.set_value_type(results[0], ty);
                    }
                } else {
                    self.expect_token(Token::LeftParen)?;
                    for (i, &result) in results.iter().enumerate() {
                        if i > 0 {
                            self.expect_token(Token::Comma)?;
                        }
                        let ty = self.parse_type()?;
                        if let Some(_region) = self.current_region {
                            self.ctx.set_value_type(result, ty);
                        }
                    }
                    self.expect_token(Token::RightParen)?;
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
        while !self.is_at_end() {
            if self.is_at_end() {
                break;
            }
            
            self.parse_operation()?;
        }
        
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