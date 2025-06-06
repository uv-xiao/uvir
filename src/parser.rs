use crate::error::{Error, Result};

pub struct Parser {
    input: String,
    position: usize,
}

impl Parser {
    pub fn new(input: String) -> Self {
        Self {
            input,
            position: 0,
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
}