use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // Literals
    #[regex(r"-?[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    IntegerLiteral(i64),

    #[regex(r"0x[0-9a-fA-F]+", |lex| i64::from_str_radix(&lex.slice()[2..], 16).ok())]
    HexIntegerLiteral(i64),

    #[regex(r"[-+]?[0-9]+\.[0-9]*([eE][-+]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok())]
    FloatLiteral(f64),

    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        // Remove surrounding quotes and handle basic escapes 
        let inner = &s[1..s.len()-1];
        let mut result = String::new();
        let mut chars = inner.chars();
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some(c) => {
                        result.push('\\');
                        result.push(c);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(ch);
            }
        }
        Some(result)
    })]
    StringLiteral(String),

    // Identifiers
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_$]*", |lex| lex.slice().to_string())]
    BareId(String),

    // Value references
    #[regex(r"%[0-9]+", |lex| {
        let s = &lex.slice()[1..]; // Skip the %
        s.parse::<u64>().ok().map(|n| format!("{}", n))
    })]
    ValueId(String),

    #[regex(r"%[a-zA-Z_$][a-zA-Z0-9_$]*", |lex| {
        let s = &lex.slice()[1..]; // Skip the %
        s.to_string()
    })]
    NamedValueId(String),

    // Symbol references
    #[regex(r"@([a-zA-Z_$][a-zA-Z0-9_$]*|[0-9]+)", |lex| {
        let s = &lex.slice()[1..]; // Skip the @
        s.to_string()
    })]
    SymbolRef(String),

    // Punctuation
    #[token("(")]
    LeftParen,

    #[token(")")]
    RightParen,

    #[token("{")]
    LeftBrace,

    #[token("}")]
    RightBrace,

    #[token("[")]
    LeftBracket,

    #[token("]")]
    RightBracket,

    #[token("<")]
    LeftAngle,

    #[token(">")]
    RightAngle,

    #[token(",")]
    Comma,

    #[token(":")]
    Colon,

    #[token("::")]
    DoubleColon,

    #[token("=")]
    Equals,

    #[token(".")]
    Dot,

    #[token("->")]
    Arrow,

    #[token("!")]
    Bang,

    #[token("#")]
    Hash,

    #[token("^")]
    Caret,

    // Keywords (built-in types)
    #[token("i1")]
    I1,
    #[token("i8")]
    I8,
    #[token("i16")]
    I16,
    #[token("i32")]
    I32,
    #[token("i64")]
    I64,
    #[token("u8")]
    U8,
    #[token("u16")]
    U16,
    #[token("u32")]
    U32,
    #[token("u64")]
    U64,
    #[token("f16")]
    F16,
    #[token("f32")]
    F32,
    #[token("f64")]
    F64,
    #[token("bf16")]
    BF16,
    #[token("index")]
    Index,

    // More builtin types
    #[token("none")]
    None,

    #[token("complex")]
    Complex,

    #[token("memref")]
    Memref,

    #[token("tensor")]
    Tensor,

    #[token("vector")]
    Vector,

    // Generic integer/unsigned types
    #[regex(r"i[0-9]+", |lex| {
        let s = &lex.slice()[1..];
        s.parse::<u32>().ok()
    })]
    GenericInteger(u32),

    #[regex(r"u[0-9]+", |lex| {
        let s = &lex.slice()[1..];
        s.parse::<u32>().ok()
    })]
    GenericUnsigned(u32),

    #[regex(r"si[0-9]+", |lex| {
        let s = &lex.slice()[2..];
        s.parse::<u32>().ok()
    })]
    GenericSigned(u32),

    // Keywords
    #[token("module")]
    Module,

    #[token("func")]
    Func,

    #[token("return")]
    Return,

    #[token("loc")]
    Loc,

    #[token("true")]
    True,

    #[token("false")]
    False,

    #[token("unit")]
    Unit,

    #[token("dense")]
    Dense,

    #[token("sparse")]
    Sparse,

    #[token("affine_map")]
    AffineMap,

    #[token("affine_set")]
    AffineSet,

    // Special symbols
    #[token("?")]
    Question,

    #[token("*")]
    Star,

    #[token("x")]
    X,

    // Dimension specification tokens for tensor/vector/memref shapes
    // Matches patterns like "4x8xf32", "?x10xf32", "*x", etc.
    #[regex(r"(\?|[0-9]+|\*)x", |lex| lex.slice().to_string())]
    DimensionPrefix(String),

    // Comments and whitespace (skip)
    #[regex(r"//[^\r\n]*", logos::skip)]
    #[regex(r"[ \t\r\n\f]+", logos::skip)]
    Whitespace,

    // Error token
    Error,
}

impl Token {
    /// Returns true if this token represents a type
    pub fn is_type(&self) -> bool {
        matches!(
            self,
            Token::I1
                | Token::I8
                | Token::I16
                | Token::I32
                | Token::I64
                | Token::U8
                | Token::U16
                | Token::U32
                | Token::U64
                | Token::F16
                | Token::F32
                | Token::F64
                | Token::BF16
                | Token::Index
                | Token::None
                | Token::Complex
                | Token::Memref
                | Token::Tensor
                | Token::Vector
                | Token::GenericInteger(_)
                | Token::GenericUnsigned(_)
                | Token::GenericSigned(_)
                | Token::Bang // For dialect types starting with !
        )
    }

    /// Returns true if this token can start an identifier
    pub fn is_identifier(&self) -> bool {
        matches!(self, Token::BareId(_))
    }

    /// Returns true if this token represents a value reference
    pub fn is_value_ref(&self) -> bool {
        matches!(self, Token::ValueId(_) | Token::NamedValueId(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lex = Token::lexer("func.call %0, %arg1 : (i32, i32) -> i32");

        assert_eq!(lex.next(), Some(Ok(Token::Func)));
        assert_eq!(lex.next(), Some(Ok(Token::Dot)));
        assert_eq!(lex.next(), Some(Ok(Token::BareId("call".to_string()))));
        assert_eq!(lex.next(), Some(Ok(Token::ValueId("0".to_string()))));
        assert_eq!(lex.next(), Some(Ok(Token::Comma)));
        assert_eq!(
            lex.next(),
            Some(Ok(Token::NamedValueId("arg1".to_string())))
        );
        assert_eq!(lex.next(), Some(Ok(Token::Colon)));
        assert_eq!(lex.next(), Some(Ok(Token::LeftParen)));
        assert_eq!(lex.next(), Some(Ok(Token::I32)));
        assert_eq!(lex.next(), Some(Ok(Token::Comma)));
        assert_eq!(lex.next(), Some(Ok(Token::I32)));
        assert_eq!(lex.next(), Some(Ok(Token::RightParen)));
        assert_eq!(lex.next(), Some(Ok(Token::Arrow)));
        assert_eq!(lex.next(), Some(Ok(Token::I32)));
    }

    #[test]
    fn test_literals() {
        let mut lex = Token::lexer(r#"42 -123 0xFF 3.14 "hello world""#);

        assert_eq!(lex.next(), Some(Ok(Token::IntegerLiteral(42))));
        assert_eq!(lex.next(), Some(Ok(Token::IntegerLiteral(-123))));
        assert_eq!(lex.next(), Some(Ok(Token::HexIntegerLiteral(255))));
        assert_eq!(lex.next(), Some(Ok(Token::FloatLiteral(3.14))));
        assert_eq!(
            lex.next(),
            Some(Ok(Token::StringLiteral("hello world".to_string())))
        );
    }

    #[test]
    fn test_generic_types() {
        let mut lex = Token::lexer("i128 u256");

        assert_eq!(lex.next(), Some(Ok(Token::GenericInteger(128))));
        assert_eq!(lex.next(), Some(Ok(Token::GenericUnsigned(256))));
    }

    #[test]
    fn test_comments_ignored() {
        let mut lex = Token::lexer("i32 // this is a comment\ni64");

        assert_eq!(lex.next(), Some(Ok(Token::I32)));
        assert_eq!(lex.next(), Some(Ok(Token::I64)));
        assert_eq!(lex.next(), None);
    }
}
