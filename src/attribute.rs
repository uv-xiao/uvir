use crate::string_interner::StringId;
use crate::types::TypeId;
use smallvec::SmallVec;
use std::any;

#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Integer(i64),
    Float(f64),
    String(StringId),
    Type(TypeId),
    Array(Vec<Attribute>),
    Dialect {
        dialect: StringId,
        data: AttributeStorage,
    },
}

pub struct AttributeStorage {
    data: SmallVec<[u8; 24]>,
    vtable: &'static AttributeVTable,
}

impl AttributeStorage {
    pub fn new<T: DialectAttribute>(value: T) -> Self {
        let mut data = SmallVec::new();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &value as *const T as *const u8,
                std::mem::size_of::<T>(),
            )
        };
        data.extend_from_slice(bytes);
        std::mem::forget(value);

        Self {
            data,
            vtable: T::vtable(),
        }
    }

    pub fn as_ref<T: DialectAttribute>(&self) -> Option<&T> {
        if self.vtable.type_id == any::TypeId::of::<T>() {
            Some(unsafe { &*(self.data.as_ptr() as *const T) })
        } else {
            None
        }
    }
}

impl Clone for AttributeStorage {
    fn clone(&self) -> Self {
        Self {
            data: (self.vtable.clone)(&self.data),
            vtable: self.vtable,
        }
    }
}

impl PartialEq for AttributeStorage {
    fn eq(&self, other: &Self) -> bool {
        if self.vtable as *const _ != other.vtable as *const _ {
            return false;
        }
        (self.vtable.eq)(&self.data, &other.data)
    }
}

impl std::fmt::Debug for AttributeStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttributeStorage")
            .field("type", &self.vtable.name)
            .finish()
    }
}

impl Drop for AttributeStorage {
    fn drop(&mut self) {
        if let Some(drop_fn) = self.vtable.drop_fn {
            drop_fn(&mut self.data);
        }
    }
}

pub struct AttributeVTable {
    pub type_id: any::TypeId,
    pub name: &'static str,
    pub clone: fn(&[u8]) -> SmallVec<[u8; 24]>,
    pub eq: fn(&[u8], &[u8]) -> bool,
    pub drop_fn: Option<fn(&mut SmallVec<[u8; 24]>)>,
}

pub trait DialectAttribute: Sized + Clone + PartialEq + 'static {
    fn vtable() -> &'static AttributeVTable;
}

pub type AttributeMap = SmallVec<[(StringId, Attribute); 4]>;

pub trait AttributeMapExt {
    fn get(&self, key: StringId) -> Option<&Attribute>;
    fn insert(&mut self, key: StringId, value: Attribute);
    fn remove(&mut self, key: StringId) -> Option<Attribute>;
}

impl AttributeMapExt for AttributeMap {
    fn get(&self, key: StringId) -> Option<&Attribute> {
        self.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    fn insert(&mut self, key: StringId, value: Attribute) {
        if let Some(pos) = self.iter().position(|(k, _)| *k == key) {
            self[pos].1 = value;
        } else {
            self.push((key, value));
        }
    }

    fn remove(&mut self, key: StringId) -> Option<Attribute> {
        if let Some(pos) = self.iter().position(|(k, _)| *k == key) {
            Some(self.remove(pos).1)
        } else {
            None
        }
    }
}

#[macro_export]
macro_rules! impl_dialect_attribute {
    ($type:ty) => {
        impl $crate::attribute::DialectAttribute for $type {
            fn vtable() -> &'static $crate::attribute::AttributeVTable {
                use std::any::TypeId;
                use smallvec::SmallVec;
                
                static VTABLE: std::sync::OnceLock<$crate::attribute::AttributeVTable> = std::sync::OnceLock::new();
                VTABLE.get_or_init(|| $crate::attribute::AttributeVTable {
                    type_id: TypeId::of::<$type>(),
                    name: stringify!($type),
                    clone: |data| {
                        let value = unsafe { &*(data.as_ptr() as *const $type) };
                        let cloned = value.clone();
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                &cloned as *const $type as *const u8,
                                std::mem::size_of::<$type>(),
                            )
                        };
                        let mut result = SmallVec::new();
                        result.extend_from_slice(bytes);
                        std::mem::forget(cloned);
                        result
                    },
                    eq: |a, b| {
                        let a = unsafe { &*(a.as_ptr() as *const $type) };
                        let b = unsafe { &*(b.as_ptr() as *const $type) };
                        a == b
                    },
                    drop_fn: if std::mem::needs_drop::<$type>() {
                        Some(|data| {
                            unsafe {
                                std::ptr::drop_in_place(data.as_mut_ptr() as *mut $type);
                            }
                        })
                    } else {
                        None
                    },
                })
            }
        }
    };
}