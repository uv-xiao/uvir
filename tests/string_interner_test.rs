// Tests for the string interning functionality in uvir.
// 
// Purpose: Validates that the string interner correctly:
// - Deduplicates identical strings to save memory
// - Handles Unicode and special characters properly
// - Maintains string identity across operations
// - Scales efficiently with many unique strings
//
// This is a core feature as string interning is used throughout the IR
// for identifiers, operation names, and string attributes.

use uvir::Context;

#[test]
fn test_basic_string_interning() {
    let mut ctx = Context::new();
    
    let id1 = ctx.intern_string("hello");
    let id2 = ctx.intern_string("world");
    let id3 = ctx.intern_string("hello");
    
    // Same string should get same ID
    assert_eq!(id1, id3);
    assert_ne!(id1, id2);
    
    // Should be able to retrieve strings
    assert_eq!(ctx.get_string(id1), Some("hello"));
    assert_eq!(ctx.get_string(id2), Some("world"));
}

#[test]
fn test_empty_string_interning() {
    let mut ctx = Context::new();
    
    let empty_id = ctx.intern_string("");
    let empty_id2 = ctx.intern_string("");
    
    assert_eq!(empty_id, empty_id2);
    assert_eq!(ctx.get_string(empty_id), Some(""));
}

#[test]
fn test_unicode_string_interning() {
    let mut ctx = Context::new();
    
    let id1 = ctx.intern_string("Hello 世界 🌍");
    let id2 = ctx.intern_string("Hello 世界 🌍");
    let id3 = ctx.intern_string("Different 字符串");
    
    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
    
    assert_eq!(ctx.get_string(id1), Some("Hello 世界 🌍"));
    assert_eq!(ctx.get_string(id3), Some("Different 字符串"));
}

#[test]
fn test_many_unique_strings() {
    let mut ctx = Context::new();
    let mut ids = Vec::new();
    
    // Intern many unique strings
    for i in 0..1000 {
        let s = format!("string_{}", i);
        ids.push((ctx.intern_string(&s), s));
    }
    
    // Verify all strings can be retrieved correctly
    for (id, expected) in &ids {
        assert_eq!(ctx.get_string(*id), Some(expected.as_str()));
    }
    
    // Verify re-interning gives same IDs
    for (expected_id, s) in &ids[..10] {
        let id = ctx.intern_string(s);
        assert_eq!(id, *expected_id);
    }
}

#[test]
fn test_string_id_properties() {
    let mut ctx = Context::new();
    
    let id1 = ctx.intern_string("first");
    let id2 = ctx.intern_string("second");
    
    // StringId should be copyable
    let id1_copy = id1;
    assert_eq!(id1, id1_copy);
    
    // StringId should be hashable (compile-time check)
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert(id1, "value1");
    map.insert(id2, "value2");
    
    assert_eq!(map.get(&id1), Some(&"value1"));
    assert_eq!(map.get(&id2), Some(&"value2"));
}

#[test]
fn test_special_characters() {
    let mut ctx = Context::new();
    
    let strings = vec![
        "with\nnewline",
        "with\ttab",
        "with\"quote",
        "with\\backslash",
        "with\0null",
        "   leading_spaces",
        "trailing_spaces   ",
        "   both_spaces   ",
    ];
    
    let mut ids = Vec::new();
    for s in &strings {
        ids.push(ctx.intern_string(s));
    }
    
    // All should be different
    for i in 0..ids.len() {
        for j in i+1..ids.len() {
            assert_ne!(ids[i], ids[j], "Strings '{}' and '{}' should have different IDs", 
                      strings[i], strings[j]);
        }
    }
    
    // All should be retrievable
    for (id, s) in ids.iter().zip(&strings) {
        assert_eq!(ctx.get_string(*id), Some(*s));
    }
}

#[test]
fn test_string_interner_deduplication() {
    let mut ctx = Context::new();
    
    // Test that interning the same string multiple times doesn't grow memory
    let first_id = ctx.intern_string("duplicate_me");
    
    // Intern the same string 1000 times
    for _ in 0..1000 {
        let id = ctx.intern_string("duplicate_me");
        assert_eq!(id, first_id);
    }
    
    // The string should still be retrievable
    assert_eq!(ctx.get_string(first_id), Some("duplicate_me"));
}