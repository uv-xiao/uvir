use crate::string_interner::StringId;
use crate::parser::Parser;
use crate::printer::Printer;
use crate::error::Result;
use smallvec::SmallVec;
use std::any;
use std::fmt;
use ahash::AHashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TypeId(u32);

impl TypeId {
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Integer { width: u32, signed: bool },
    Float { precision: FloatPrecision },
    Function { inputs: Vec<TypeId>, outputs: Vec<TypeId> },
    Dialect { 
        dialect: StringId,
        data: TypeStorage,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatPrecision {
    Half,
    Single,
    Double,
}

pub struct TypeStorage {
    data: SmallVec<[u8; 16]>,
    vtable: &'static TypeVTable,
}

impl TypeStorage {
    pub fn new<T: DialectType>(value: T) -> Self {
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

    pub fn as_ref<T: DialectType>(&self) -> Option<&T> {
        if self.vtable.type_id == any::TypeId::of::<T>() {
            Some(unsafe { &*(self.data.as_ptr() as *const T) })
        } else {
            None
        }
    }
}

impl Clone for TypeStorage {
    fn clone(&self) -> Self {
        Self {
            data: (self.vtable.clone)(&self.data),
            vtable: self.vtable,
        }
    }
}

impl PartialEq for TypeStorage {
    fn eq(&self, other: &Self) -> bool {
        if self.vtable as *const _ != other.vtable as *const _ {
            return false;
        }
        (self.vtable.eq)(&self.data, &other.data)
    }
}

impl Eq for TypeStorage {}

impl std::hash::Hash for TypeStorage {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.vtable as *const _ as usize).hash(state);
        self.data.hash(state);
    }
}

impl fmt::Debug for TypeStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypeStorage")
            .field("type", &self.vtable.name)
            .finish()
    }
}

impl Drop for TypeStorage {
    fn drop(&mut self) {
        if let Some(drop_fn) = self.vtable.drop_fn {
            drop_fn(&mut self.data);
        }
    }
}

pub struct TypeVTable {
    pub type_id: any::TypeId,
    pub name: &'static str,
    pub parse: fn(&mut Parser) -> Result<TypeStorage>,
    pub print: fn(&[u8], &mut Printer) -> Result<()>,
    pub clone: fn(&[u8]) -> SmallVec<[u8; 16]>,
    pub eq: fn(&[u8], &[u8]) -> bool,
    pub drop_fn: Option<fn(&mut SmallVec<[u8; 16]>)>,
}

pub trait DialectType: Sized + Clone + PartialEq + 'static {
    fn vtable() -> &'static TypeVTable;
    fn parse(parser: &mut Parser) -> Result<Self>;
    fn print(&self, printer: &mut Printer) -> Result<()>;
}

pub struct TypeInterner {
    types: Vec<TypeKind>,
    lookup: AHashMap<TypeKind, TypeId>,
}

impl TypeInterner {
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            lookup: AHashMap::new(),
        }
    }

    pub fn intern(&mut self, kind: TypeKind) -> TypeId {
        if let Some(&id) = self.lookup.get(&kind) {
            return id;
        }

        let id = TypeId(self.types.len() as u32);
        self.types.push(kind.clone());
        self.lookup.insert(kind, id);
        id
    }

    pub fn get(&self, id: TypeId) -> Option<&TypeKind> {
        self.types.get(id.0 as usize)
    }

    pub fn get_unchecked(&self, id: TypeId) -> &TypeKind {
        &self.types[id.0 as usize]
    }
}

impl Default for TypeInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! impl_dialect_type {
    ($type:ty) => {
        impl $crate::types::DialectType for $type {
            fn vtable() -> &'static $crate::types::TypeVTable {
                use std::any::TypeId;
                use smallvec::SmallVec;
                
                static VTABLE: std::sync::OnceLock<$crate::types::TypeVTable> = std::sync::OnceLock::new();
                VTABLE.get_or_init(|| $crate::types::TypeVTable {
                    type_id: TypeId::of::<$type>(),
                    name: stringify!($type),
                    parse: |parser| {
                        <$type>::parse(parser).map($crate::types::TypeStorage::new)
                    },
                    print: |data, printer| {
                        let value = unsafe { &*(data.as_ptr() as *const $type) };
                        value.print(printer)
                    },
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
            
            fn parse(parser: &mut $crate::parser::Parser) -> $crate::error::Result<Self> {
                Self::parse(parser)
            }
            
            fn print(&self, printer: &mut $crate::printer::Printer) -> $crate::error::Result<()> {
                self.print(printer)
            }
        }
    };
}