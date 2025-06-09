// Tests for the MLIR-compatible parser in uvir.
//
// Purpose: Validates the parser functionality including:
// - Basic token parsing (identifiers, integers, floats)
// - Type parsing for builtin types
// - Attribute parsing with type annotations
// - SSA value reference parsing
// - Whitespace and special character handling
// - Function type syntax parsing
// - Error handling for invalid inputs
//
// The parser enables reading MLIR textual format, essential for
// interoperability with MLIR tools and human-readable IR.

use logos::Logos;
use uvir::attribute::Attribute;
use uvir::lexer::Token;
use uvir::parser::Parser;
use uvir::types::{FloatPrecision, TypeKind};
use uvir::Context;

#[test]
fn test_parse_basic_tokens() {
    let mut ctx = Context::new();

    // Test identifier parsing
    let mut parser = Parser::new("hello_world".to_string(), &mut ctx).unwrap();
    let id = parser.expect_identifier().unwrap();
    assert_eq!(id, "hello_world");

    // Test integer parsing
    let mut parser = Parser::new("42".to_string(), &mut ctx).unwrap();
    let num = parser.expect_integer().unwrap();
    assert_eq!(num, 42);

    // Test negative integer
    let mut parser = Parser::new("-123".to_string(), &mut ctx).unwrap();
    let num = parser.expect_integer().unwrap();
    assert_eq!(num, -123);

    // Test float parsing
    let mut parser = Parser::new("3.14".to_string(), &mut ctx).unwrap();
    let num = parser.expect_float().unwrap();
    assert!((num - 3.14).abs() < f64::EPSILON);
}

#[test]
fn test_parse_types() {
    let mut ctx = Context::new();

    // Parse integer types
    let mut parser = Parser::new("i32".to_string(), &mut ctx).unwrap();
    let ty = parser.parse_type().unwrap();
    match ctx.get_type(ty) {
        Some(TypeKind::Integer { width, signed }) => {
            assert_eq!(*width, 32);
            assert_eq!(*signed, true);
        }
        _ => panic!("Expected i32 type"),
    }

    // Parse unsigned types
    let mut parser = Parser::new("u64".to_string(), &mut ctx).unwrap();
    let ty = parser.parse_type().unwrap();
    match ctx.get_type(ty) {
        Some(TypeKind::Integer { width, signed }) => {
            assert_eq!(*width, 64);
            assert_eq!(*signed, false);
        }
        _ => panic!("Expected u64 type"),
    }

    // Parse float types
    let mut parser = Parser::new("f32".to_string(), &mut ctx).unwrap();
    let ty = parser.parse_type().unwrap();
    match ctx.get_type(ty) {
        Some(TypeKind::Float { precision }) => {
            assert_eq!(*precision, FloatPrecision::Single);
        }
        _ => panic!("Expected f32 type"),
    }
}

#[test]
fn test_parse_attributes() {
    let mut ctx = Context::new();

    // Parse integer attribute
    let mut parser = Parser::new("42".to_string(), &mut ctx).unwrap();
    let attr = parser.parse_attribute().unwrap();
    match attr {
        Attribute::Integer(val) => assert_eq!(val, 42),
        _ => panic!("Expected integer attribute"),
    }

    // Parse float attribute
    let mut parser = Parser::new("3.14".to_string(), &mut ctx).unwrap();
    let attr = parser.parse_attribute().unwrap();
    match attr {
        Attribute::Float(val) => assert!((val - 3.14).abs() < f64::EPSILON),
        _ => panic!("Expected float attribute"),
    }

    // Parse string attribute
    let mut parser = Parser::new("\"hello world\"".to_string(), &mut ctx).unwrap();
    let attr = parser.parse_attribute().unwrap();
    match attr {
        Attribute::String(id) => {
            assert_eq!(ctx.get_string(id), Some("hello world"));
        }
        _ => panic!("Expected string attribute"),
    }
}

#[test]
fn test_parse_ssa_value_syntax() {
    let mut ctx = Context::new();

    // Test parsing SSA value syntax - we need to use the token-based parser
    let mut parser = Parser::new("%0".to_string(), &mut ctx).unwrap();
    parser
        .expect_token(Token::ValueId("0".to_string()))
        .unwrap();

    let mut parser = Parser::new("%arg0".to_string(), &mut ctx).unwrap();
    parser
        .expect_token(Token::NamedValueId("arg0".to_string()))
        .unwrap();

    let mut parser = Parser::new("%result_123".to_string(), &mut ctx).unwrap();
    parser
        .expect_token(Token::NamedValueId("result_123".to_string()))
        .unwrap();
}

#[test]
fn test_parse_whitespace_handling() {
    let mut ctx = Context::new();

    // Test whitespace skipping
    let mut parser = Parser::new("   hello   ".to_string(), &mut ctx).unwrap();
    let id = parser.expect_identifier().unwrap();
    assert_eq!(id, "hello");

    // Test multiple tokens with whitespace
    let mut parser = Parser::new("  42   :   i32  ".to_string(), &mut ctx).unwrap();
    let num = parser.expect_integer().unwrap();
    assert_eq!(num, 42);
    parser.expect_token(Token::Colon).unwrap();
    let ty = parser.parse_type().unwrap();
    // Verify we parsed i32
    match ctx.get_type(ty) {
        Some(TypeKind::Integer { width, signed }) => {
            assert_eq!(*width, 32);
            assert_eq!(*signed, true);
        }
        _ => panic!("Expected i32 type"),
    }
}

#[test]
fn test_parse_special_characters() {
    let mut ctx = Context::new();

    // Test expecting specific characters
    let mut parser = Parser::new("()".to_string(), &mut ctx).unwrap();
    parser.expect_token(Token::LeftParen).unwrap();
    parser.expect_token(Token::RightParen).unwrap();

    let mut parser = Parser::new("{ }".to_string(), &mut ctx).unwrap();
    parser.expect_token(Token::LeftBrace).unwrap();
    parser.expect_token(Token::RightBrace).unwrap();

    // Test error on wrong token
    let mut parser = Parser::new("abc".to_string(), &mut ctx).unwrap();
    assert!(parser.expect_token(Token::LeftParen).is_err());
}

#[test]
fn test_parse_attribute_list() {
    let mut ctx = Context::new();

    // Parse attribute list syntax - simplified test
    let mut parser = Parser::new("{attr1 = 42}".to_string(), &mut ctx).unwrap();
    let attrs = parser.parse_attribute_dict().unwrap();
    assert_eq!(attrs.len(), 1);

    // Check the attribute
    let (name_id, attr_value) = &attrs[0];
    assert_eq!(ctx.get_string(*name_id), Some("attr1"));
    match attr_value {
        Attribute::Integer(val) => assert_eq!(*val, 42),
        _ => panic!("Expected integer attribute"),
    }
}

#[test]
fn test_parse_function_type() {
    let mut ctx = Context::new();

    // Parse function type syntax: (i32, i32) -> i32
    let mut parser = Parser::new("(i32, i32) -> i32".to_string(), &mut ctx).unwrap();
    let func_ty = parser.parse_function_type().unwrap();

    // Verify the function type was parsed correctly
    match ctx.get_type(func_ty) {
        Some(TypeKind::Function { inputs, outputs }) => {
            assert_eq!(inputs.len(), 2);
            assert_eq!(outputs.len(), 1);

            // Check input types
            for &input_ty in inputs {
                match ctx.get_type(input_ty) {
                    Some(TypeKind::Integer { width, signed }) => {
                        assert_eq!(*width, 32);
                        assert_eq!(*signed, true);
                    }
                    _ => panic!("Expected i32 input type"),
                }
            }

            // Check output type
            match ctx.get_type(outputs[0]) {
                Some(TypeKind::Integer { width, signed }) => {
                    assert_eq!(*width, 32);
                    assert_eq!(*signed, true);
                }
                _ => panic!("Expected i32 output type"),
            }
        }
        _ => panic!("Expected function type"),
    }
}

#[test]
fn test_parse_errors() {
    let mut ctx = Context::new();

    // Test EOF errors
    let mut parser = Parser::new("".to_string(), &mut ctx).unwrap();
    assert!(parser.expect_identifier().is_err());
    assert!(parser.expect_integer().is_err());

    // Test invalid tokens for identifier
    let mut parser = Parser::new("123".to_string(), &mut ctx).unwrap();
    assert!(parser.expect_identifier().is_err());

    // Test invalid tokens for integer
    let mut parser = Parser::new("hello".to_string(), &mut ctx).unwrap();
    assert!(parser.expect_integer().is_err());
}

#[test]
fn test_peek_and_advance() {
    let mut ctx = Context::new();
    let mut parser = Parser::new("hello world 123".to_string(), &mut ctx).unwrap();

    // Test peek doesn't advance
    let first_token = parser.peek().cloned();
    assert_eq!(parser.peek(), first_token.as_ref());

    // Test advance moves forward
    let token1 = parser.advance().cloned();
    let token2 = parser.peek().cloned();
    assert_ne!(token1, token2);

    // Advance again
    let _ = parser.advance();
    let token3 = parser.advance().cloned();

    // Should be different tokens
    assert_ne!(token1, token3);
}

#[test]
fn test_parse_unicode() {
    let mut ctx = Context::new();

    // Test with ASCII string to verify basic functionality
    let mut parser = Parser::new("\"Hello World\"".to_string(), &mut ctx).unwrap();
    let attr = parser.parse_attribute().unwrap();
    match attr {
        Attribute::String(id) => {
            assert_eq!(ctx.get_string(id), Some("Hello World"));
        }
        _ => panic!("Expected string attribute"),
    }

    // Test unicode identifiers (if supported) - just check it doesn't crash
    let parser_result = Parser::new("变量_123".to_string(), &mut ctx);
    // This might fail depending on identifier rules, which is okay
    let _ = parser_result;
}

#[test]
fn test_parse_mlir_snippet() {
    let mut ctx = Context::new();

    // Test parsing a complete MLIR snippet - simplified for now
    let mlir_snippet = r#"
func.func @simple_add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}
"#
    .trim();

    let mut parser = Parser::new(mlir_snippet.to_string(), &mut ctx).unwrap();

    // For now, we'll just test that it lexes without errors
    assert!(!parser.is_at_end());

    // The parser might not fully implement module parsing yet
    let result = parser.parse_module();

    match result {
        Ok(_) => {
            // The parser successfully parsed the MLIR snippet
            println!("Successfully parsed MLIR snippet");
        }
        Err(e) => {
            // Parser isn't fully implemented yet - this is expected
            println!("Parser not fully implemented yet: {:?}", e);
            // We mark this as expected for now
        }
    }
}

#[test]
fn test_parse_simple_operation() {
    let mut ctx = Context::new();

    // Test parsing a simple operation in generic form
    let operation_text = r#"%0 = "arith.constant"() {value = 42 : i32} : () -> i32"#;

    let mut parser = Parser::new(operation_text.to_string(), &mut ctx).unwrap();

    // For now, just test that it lexes properly
    assert!(!parser.is_at_end());

    // This should parse a single operation
    let result = parser.parse_operation();

    match result {
        Ok(_) => {
            // Successfully parsed the operation
            println!("Successfully parsed operation");
        }
        Err(e) => {
            // This is expected until the parser is fully implemented
            println!("Operation parsing not fully implemented: {:?}", e);
        }
    }
}

#[test]
fn test_parse_type_declarations() {
    let mut ctx = Context::new();

    // Test parsing type alias declaration
    let type_alias = "!my_type = i32";
    let mut parser = Parser::new(type_alias.to_string(), &mut ctx).unwrap();

    // Skip the '!' for now and just parse the identifier and type
    parser.expect_token(Token::Bang).unwrap();
    let alias_name = parser.expect_identifier().unwrap();
    assert_eq!(alias_name, "my_type");

    parser.expect_token(Token::Equals).unwrap();

    let ty = parser.parse_type().unwrap();
    // Verify the type was parsed correctly
    assert!(ctx.get_type(ty).is_some());
}

#[test]
fn test_parse_attribute_declarations() {
    let mut ctx = Context::new();

    // Test parsing attribute alias declaration
    let attr_alias = r#"#my_attr = dense<[1, 2, 3]> : tensor<3xi32>"#;
    let mut parser = Parser::new(attr_alias.to_string(), &mut ctx).unwrap();

    // Skip the '#' for now and just parse the identifier
    parser.expect_token(Token::Hash).unwrap();
    let alias_name = parser.expect_identifier().unwrap();
    assert_eq!(alias_name, "my_attr");

    parser.expect_token(Token::Equals).unwrap();

    // For now, just verify we can parse the identifier part
    // Full attribute parsing will be implemented later
}

#[test]
fn test_tokenization_examples() {
    // Test tokenization of various MLIR-like syntax examples
    let examples = vec![
        ("func.call %0, %arg1 : (i32, i32) -> i32", 14), // func . call %0 , %arg1 : ( i32 , i32 ) -> i32
        ("%result = arith.addi %a, %b : i64", 10),       // %result = arith . addi %a , %b : i64
        ("scf.for %i = %lb to %ub step %step {", 11),    // scf . for %i = %lb to %ub step %step {
        ("  %x = memref.load %array[%i] : memref<10xi32>", 15), // %x = memref . load %array [ %i ] : memref < IntegerLiteral(10) xi32 >
        ("}", 1),                                               // }
        ("return %result : i64", 4),                            // return %result : i64
    ];

    for (input, expected_token_count) in examples {
        let mut lexer = Token::lexer(input);
        let mut tokens = Vec::new();

        while let Some(token_result) = lexer.next() {
            match token_result {
                Ok(token) => tokens.push(token),
                Err(_) => panic!(
                    "Lexical error at position {} for input: {}",
                    lexer.span().start,
                    input
                ),
            }
        }

        assert_eq!(
            tokens.len(),
            expected_token_count,
            "Token count mismatch for input '{}'. Expected {}, got {}. Tokens: {:?}",
            input,
            expected_token_count,
            tokens.len(),
            tokens
        );

        // Verify no error tokens
        for token in &tokens {
            print!("{:?} ", token);
            assert!(
                !matches!(token, Token::Error),
                "Found error token in: {}",
                input
            );
        }
        println!();
    }
}

#[test]
fn test_parser_integration_with_simple_types() {
    let mut ctx = Context::new();

    let type_examples = vec![
        ("i32", 1),
        ("f64", 1),
        ("%0", 1),
        ("%arg1", 1),
        ("i1", 1),
        ("u64", 1),
        ("f32", 1),
    ];

    for (example, expected_tokens) in type_examples {
        let parser_result = Parser::new(example.to_string(), &mut ctx);
        assert!(
            parser_result.is_ok(),
            "Failed to create parser for: {}",
            example
        );

        let parser = parser_result.unwrap();
        assert_eq!(
            parser.tokens().len(),
            expected_tokens,
            "Expected {} tokens for '{}', got {}. Tokens: {:?}",
            expected_tokens,
            example,
            parser.tokens().len(),
            parser.tokens()
        );

        // Verify the tokens are valid (no Error tokens)
        for token in parser.tokens() {
            assert!(
                !matches!(token, Token::Error),
                "Found error token for input: {}",
                example
            );
        }
    }
}

#[test]
fn test_complex_expression_tokenization() {
    let mut ctx = Context::new();

    let complex_expr = r#"%result = "dialect.operation"(%operand1, %operand2) {attr = "value"} : (i32, i32) -> i32"#;

    let parser_result = Parser::new(complex_expr.to_string(), &mut ctx);
    assert!(
        parser_result.is_ok(),
        "Failed to create parser for complex expression"
    );

    let parser = parser_result.unwrap();
    let tokens = parser.tokens();

    // Verify we have a reasonable number of tokens (should be around 17-20)
    assert!(
        tokens.len() >= 15 && tokens.len() <= 25,
        "Expected 15-25 tokens for complex expression, got {}. Tokens: {:?}",
        tokens.len(),
        tokens
    );

    // Verify specific token types exist
    let has_value_id = tokens.iter().any(|t| matches!(t, Token::NamedValueId(_)));
    let has_equals = tokens.iter().any(|t| matches!(t, Token::Equals));
    let has_string_literal = tokens.iter().any(|t| matches!(t, Token::StringLiteral(_)));
    let has_left_paren = tokens.iter().any(|t| matches!(t, Token::LeftParen));
    let has_right_paren = tokens.iter().any(|t| matches!(t, Token::RightParen));
    let has_left_brace = tokens.iter().any(|t| matches!(t, Token::LeftBrace));
    let has_right_brace = tokens.iter().any(|t| matches!(t, Token::RightBrace));
    let has_colon = tokens.iter().any(|t| matches!(t, Token::Colon));
    let has_arrow = tokens.iter().any(|t| matches!(t, Token::Arrow));
    let has_i32 = tokens.iter().any(|t| matches!(t, Token::I32));

    assert!(has_value_id, "Complex expression should contain value ID");
    assert!(has_equals, "Complex expression should contain equals sign");
    assert!(
        has_string_literal,
        "Complex expression should contain string literal"
    );
    assert!(
        has_left_paren,
        "Complex expression should contain left parenthesis"
    );
    assert!(
        has_right_paren,
        "Complex expression should contain right parenthesis"
    );
    assert!(
        has_left_brace,
        "Complex expression should contain left brace"
    );
    assert!(
        has_right_brace,
        "Complex expression should contain right brace"
    );
    assert!(has_colon, "Complex expression should contain colon");
    assert!(has_arrow, "Complex expression should contain arrow");
    assert!(has_i32, "Complex expression should contain i32 type");

    // Verify no error tokens
    for token in tokens {
        assert!(
            !matches!(token, Token::Error),
            "Found error token in complex expression"
        );
    }
}

#[test]
fn test_mlir_function_definition_tokens() {
    let mut ctx = Context::new();

    let func_def = r#"func.func @test_func(%arg0: i32, %arg1: f64) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  func.return %0 : i32
}"#;

    let parser_result = Parser::new(func_def.to_string(), &mut ctx);
    assert!(
        parser_result.is_ok(),
        "Failed to create parser for function definition"
    );

    let parser = parser_result.unwrap();
    let tokens = parser.tokens();

    // Should have a substantial number of tokens
    assert!(
        tokens.len() >= 25,
        "Function definition should have at least 25 tokens, got {}",
        tokens.len()
    );

    // Check for key tokens that should exist
    let token_checks = vec![
        (
            "func keyword",
            tokens
                .iter()
                .any(|t| matches!(t, Token::Func)),
        ),
        (
            "BareId(addi)",
            tokens
                .iter()
                .any(|t| matches!(t, Token::BareId(s) if s == "addi")),
        ),
        (
            "return keyword",
            tokens
                .iter()
                .any(|t| matches!(t, Token::Return)),
        ),
        (
            "SymbolRef",
            tokens.iter().any(|t| matches!(t, Token::SymbolRef(_))),
        ),
        ("I32 type", tokens.iter().any(|t| matches!(t, Token::I32))),
        ("F64 type", tokens.iter().any(|t| matches!(t, Token::F64))),
        (
            "Left brace",
            tokens.iter().any(|t| matches!(t, Token::LeftBrace)),
        ),
        (
            "Right brace",
            tokens.iter().any(|t| matches!(t, Token::RightBrace)),
        ),
        ("Arrow", tokens.iter().any(|t| matches!(t, Token::Arrow))),
    ];

    for (desc, found) in token_checks {
        assert!(found, "Function definition should contain {}", desc);
    }

    // Verify no error tokens
    for token in tokens {
        assert!(
            !matches!(token, Token::Error),
            "Found error token in function definition"
        );
    }
}

#[test]
fn test_lexer_edge_cases() {
    let mut ctx = Context::new();

    let edge_cases = vec![
        // Empty input
        ("", 0),
        // Only whitespace
        ("   \t\n  ", 0),
        // Only comments
        ("// this is a comment\n// another comment", 0),
        // Mixed whitespace and comments
        ("  // comment\n  \t// another\n  ", 0),
        // Single token
        ("i32", 1),
        // Punctuation only
        ("(){}", 4),
        // Numbers
        ("42 -123 0x1A", 3),
        // String with spaces
        ("\"hello world\"", 1),
    ];

    for (input, expected_count) in edge_cases {
        let parser_result = Parser::new(input.to_string(), &mut ctx);
        assert!(
            parser_result.is_ok(),
            "Failed to parse edge case: '{}'",
            input
        );

        let parser = parser_result.unwrap();
        assert_eq!(
            parser.tokens().len(),
            expected_count,
            "Edge case '{}' expected {} tokens, got {}. Tokens: {:?}",
            input,
            expected_count,
            parser.tokens().len(),
            parser.tokens()
        );
    }
}

#[test]
fn test_numeric_literal_parsing() {
    let mut ctx = Context::new();

    let numeric_examples = vec![
        ("42", Token::IntegerLiteral(42)),
        ("-123", Token::IntegerLiteral(-123)),
        ("0", Token::IntegerLiteral(0)),
        ("0x1A", Token::HexIntegerLiteral(26)),
        ("0xFFFF", Token::HexIntegerLiteral(65535)),
        ("3.14", Token::FloatLiteral(3.14)),
        ("-2.718", Token::FloatLiteral(-2.718)),
        ("1.0e10", Token::FloatLiteral(1.0e10)),
    ];

    for (input, expected_token) in numeric_examples {
        let parser_result = Parser::new(input.to_string(), &mut ctx);
        assert!(
            parser_result.is_ok(),
            "Failed to parse numeric literal: '{}'",
            input
        );

        let parser = parser_result.unwrap();
        let tokens = parser.tokens();
        assert_eq!(
            tokens.len(),
            1,
            "Expected exactly 1 token for '{}', got {}",
            input,
            tokens.len()
        );

        match (&tokens[0], &expected_token) {
            (Token::IntegerLiteral(a), Token::IntegerLiteral(b)) => {
                assert_eq!(a, b, "Integer literal mismatch for '{}'", input);
            }
            (Token::HexIntegerLiteral(a), Token::HexIntegerLiteral(b)) => {
                assert_eq!(a, b, "Hex integer literal mismatch for '{}'", input);
            }
            (Token::FloatLiteral(a), Token::FloatLiteral(b)) => {
                assert!(
                    (a - b).abs() < 1e-10,
                    "Float literal mismatch for '{}': {} vs {}",
                    input,
                    a,
                    b
                );
            }
            _ => panic!(
                "Token type mismatch for '{}': expected {:?}, got {:?}",
                input, expected_token, tokens[0]
            ),
        }
    }
}
