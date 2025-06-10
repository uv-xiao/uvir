// Tests for the attribute system in uvir.
//
// Purpose: Validates the attribute system functionality including:
// - Builtin attributes (integer, float, string, type, array)
// - Nested and complex attribute structures
// - Attribute maps for storing key-value pairs
// - Custom dialect attributes through type erasure
// - Attribute equality and storage efficiency
//
// Attributes are essential for storing metadata and compile-time
// constants in the IR, used by operations and optimization passes.

use uvir::attribute::AttributeStorage;
use uvir::dialects::builtin::{float_type, integer_type};
use uvir::FloatPrecision;
use uvir::{Attribute, AttributeMap, Context};

#[test]
fn test_builtin_attributes() {
    let mut ctx = Context::new();

    // Test integer attributes
    let attr1 = Attribute::Integer(42);
    let attr2 = Attribute::Integer(-100);
    let attr3 = Attribute::Integer(0);

    match attr1 {
        Attribute::Integer(val) => assert_eq!(val, 42),
        _ => panic!("Expected integer attribute"),
    }

    // Test float attributes
    let fattr1 = Attribute::Float(3.14);
    let fattr2 = Attribute::Float(-2.718);
    let fattr3 = Attribute::Float(0.0);

    match fattr1 {
        Attribute::Float(val) => assert!((val - 3.14).abs() < f64::EPSILON),
        _ => panic!("Expected float attribute"),
    }

    // Test string attributes
    let hello_id = ctx.intern_string("hello");
    let world_id = ctx.intern_string("world");

    let sattr1 = Attribute::String(hello_id);
    let sattr2 = Attribute::String(world_id);

    match sattr1 {
        Attribute::String(id) => assert_eq!(id, hello_id),
        _ => panic!("Expected string attribute"),
    }
}

#[test]
fn test_type_attributes() {
    let mut ctx = Context::new();

    let i32_type = integer_type(&mut ctx, 32, true);
    let f64_type = float_type(&mut ctx, FloatPrecision::Double);

    let type_attr1 = Attribute::Type(i32_type);
    let type_attr2 = Attribute::Type(f64_type);

    match type_attr1 {
        Attribute::Type(ty) => assert_eq!(ty, i32_type),
        _ => panic!("Expected type attribute"),
    }

    match type_attr2 {
        Attribute::Type(ty) => assert_eq!(ty, f64_type),
        _ => panic!("Expected type attribute"),
    }
}

#[test]
fn test_array_attributes() {
    let mut ctx = Context::new();
    let hello_id = ctx.intern_string("hello");

    // Empty array
    let empty_array = Attribute::Array(vec![]);

    // Homogeneous array
    let int_array = Attribute::Array(vec![
        Attribute::Integer(1),
        Attribute::Integer(2),
        Attribute::Integer(3),
    ]);

    // Heterogeneous array
    let mixed_array = Attribute::Array(vec![
        Attribute::Integer(42),
        Attribute::Float(3.14),
        Attribute::String(hello_id),
    ]);

    // Nested array
    let nested_array = Attribute::Array(vec![
        Attribute::Array(vec![Attribute::Integer(1), Attribute::Integer(2)]),
        Attribute::Array(vec![Attribute::Integer(3), Attribute::Integer(4)]),
    ]);

    match &int_array {
        Attribute::Array(elements) => {
            assert_eq!(elements.len(), 3);
            match &elements[0] {
                Attribute::Integer(val) => assert_eq!(*val, 1),
                _ => panic!("Expected integer in array"),
            }
        }
        _ => panic!("Expected array attribute"),
    }

    match &nested_array {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            match &outer[0] {
                Attribute::Array(inner) => assert_eq!(inner.len(), 2),
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected array attribute"),
    }
}

#[test]
fn test_attribute_map() {
    let mut ctx = Context::new();

    let key1 = ctx.intern_string("value");
    let key2 = ctx.intern_string("type");
    let key3 = ctx.intern_string("flags");

    let mut map: AttributeMap = AttributeMap::new();

    // Test insertion
    map.push((key1, Attribute::Integer(100)));
    map.push((key2, Attribute::String(ctx.intern_string("i32"))));
    map.push((
        key3,
        Attribute::Array(vec![
            Attribute::String(ctx.intern_string("readonly")),
            Attribute::String(ctx.intern_string("nonnull")),
        ]),
    ));

    assert_eq!(map.len(), 3);

    // Test lookup
    let found = map.iter().find(|(k, _)| *k == key1);
    assert!(found.is_some());
    match &found.unwrap().1 {
        Attribute::Integer(val) => assert_eq!(*val, 100),
        _ => panic!("Expected integer attribute"),
    }

    // Test iteration
    let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
    assert!(keys.contains(&key1));
    assert!(keys.contains(&key2));
    assert!(keys.contains(&key3));
}

#[test]
fn test_custom_dialect_attribute() {
    #[derive(Clone, PartialEq, Debug)]
    struct CustomAttribute {
        name: String,
        value: i32,
    }

    uvir::impl_dialect_attribute!(CustomAttribute);

    let mut ctx = Context::new();
    let dialect_name = ctx.intern_string("test_dialect");

    // Create custom attribute
    let custom = CustomAttribute {
        name: "test".to_string(),
        value: 100,
    };
    let storage = AttributeStorage::new(custom);

    let attr = Attribute::Dialect {
        dialect: dialect_name,
        data: storage,
    };

    // Verify we can match on it
    match &attr {
        Attribute::Dialect { dialect, data } => {
            assert_eq!(*dialect, dialect_name);
            // Verify the custom data is accessible
            assert!(data.as_ref::<CustomAttribute>().is_some());
            let custom_ref = data.as_ref::<CustomAttribute>().unwrap();
            assert_eq!(custom_ref.name, "test");
            assert_eq!(custom_ref.value, 100);
        }
        _ => panic!("Expected dialect attribute"),
    }
}

#[test]
fn test_attribute_equality() {
    let mut ctx = Context::new();

    // Integer attributes
    let int1 = Attribute::Integer(42);
    let int2 = Attribute::Integer(42);
    let int3 = Attribute::Integer(43);

    assert_eq!(int1, int2);
    assert_ne!(int1, int3);

    // Float attributes (be careful with float comparison)
    let float1 = Attribute::Float(3.14);
    let float2 = Attribute::Float(3.14);
    let float3 = Attribute::Float(2.718);

    assert_eq!(float1, float2);
    assert_ne!(float1, float3);

    // String attributes
    let str_id = ctx.intern_string("test");
    let str_id2 = ctx.intern_string("test");
    let str_id3 = ctx.intern_string("other");

    let str1 = Attribute::String(str_id);
    let str2 = Attribute::String(str_id2);
    let str3 = Attribute::String(str_id3);

    assert_eq!(str1, str2);
    assert_ne!(str1, str3);

    // Array attributes
    let arr1 = Attribute::Array(vec![Attribute::Integer(1), Attribute::Integer(2)]);
    let arr2 = Attribute::Array(vec![Attribute::Integer(1), Attribute::Integer(2)]);
    let arr3 = Attribute::Array(vec![Attribute::Integer(1), Attribute::Integer(3)]);
    let arr4 = Attribute::Array(vec![Attribute::Integer(1)]);

    assert_eq!(arr1, arr2);
    assert_ne!(arr1, arr3);
    assert_ne!(arr1, arr4);

    // Cross-type inequality
    assert_ne!(int1, float1);
    assert_ne!(int1, str1);
    assert_ne!(int1, arr1);
}

#[test]
fn test_complex_nested_attributes() {
    let mut ctx = Context::new();

    let i32_type = integer_type(&mut ctx, 32, true);
    let meta_key = ctx.intern_string("metadata");
    let name_key = ctx.intern_string("name");
    let version_key = ctx.intern_string("version");

    // Create a complex nested structure
    let complex_attr = Attribute::Array(vec![
        Attribute::Integer(42),
        Attribute::Array(vec![
            Attribute::String(meta_key),
            Attribute::Array(vec![Attribute::Type(i32_type), Attribute::Float(1.5)]),
        ]),
        Attribute::Integer(100),
    ]);

    // Verify the structure
    match &complex_attr {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 3);

            // First element
            match &outer[0] {
                Attribute::Integer(val) => assert_eq!(*val, 42),
                _ => panic!("Expected integer"),
            }

            // Second element (nested array)
            match &outer[1] {
                Attribute::Array(middle) => {
                    assert_eq!(middle.len(), 2);
                    match &middle[0] {
                        Attribute::String(id) => assert_eq!(*id, meta_key),
                        _ => panic!("Expected string"),
                    }
                    match &middle[1] {
                        Attribute::Array(inner) => {
                            assert_eq!(inner.len(), 2);
                            match &inner[0] {
                                Attribute::Type(ty) => assert_eq!(*ty, i32_type),
                                _ => panic!("Expected type"),
                            }
                        }
                        _ => panic!("Expected array"),
                    }
                }
                _ => panic!("Expected array"),
            }

            // Third element
            match &outer[2] {
                Attribute::Integer(val) => assert_eq!(*val, 100),
                _ => panic!("Expected integer"),
            }
        }
        _ => panic!("Expected array"),
    }
}

#[test]
fn test_attribute_map_operations() {
    let mut ctx = Context::new();

    let key1 = ctx.intern_string("attr1");
    let key2 = ctx.intern_string("attr2");
    let key3 = ctx.intern_string("attr3");

    let mut map: AttributeMap = AttributeMap::new();

    // Test empty map
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());

    // Add attributes
    map.push((key1, Attribute::Integer(1)));
    map.push((key2, Attribute::Integer(2)));
    map.push((key3, Attribute::Integer(3)));

    assert_eq!(map.len(), 3);
    assert!(!map.is_empty());

    // Test clear
    map.clear();
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());

    // Test with_capacity
    let mut map2 = AttributeMap::with_capacity(10);
    assert!(map2.capacity() >= 10);

    // Test multiple attributes with same key (not prevented by AttributeMap)
    map2.push((key1, Attribute::Integer(1)));
    map2.push((key1, Attribute::Integer(2)));
    assert_eq!(map2.len(), 2); // Both are stored

    let values: Vec<_> = map2
        .iter()
        .filter(|(k, _)| *k == key1)
        .map(|(_, v)| v)
        .collect();
    assert_eq!(values.len(), 2);
}
