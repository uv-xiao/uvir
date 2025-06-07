use crate::string_interner::StringId;
use crate::types::TypeId;
use crate::parser::Parser;
use crate::printer::Printer;
use crate::error::Result;
use crate::region::RegionId;
use crate::attribute::AttributeMap;
use smallvec::SmallVec;
use slotmap::new_key_type;

// Inventory collection for automatic operation registration
inventory::collect!(&'static OpInfo);

new_key_type! {
    pub struct Val;
    pub struct Opr;
}

#[derive(Clone, Copy, Debug)]
pub struct OpRef(pub Opr);

pub struct OpInfo {
    pub dialect: &'static str,
    pub name: &'static str,
    pub traits: &'static [&'static str],
    pub verify: fn(&OpData) -> Result<()>,
    pub parse: fn(&mut Parser) -> Result<OpData>,
    pub print: fn(&OpData, &mut Printer) -> Result<()>,
}

pub struct OpData {
    pub info: &'static OpInfo,
    pub operands: SmallVec<[Val; 2]>,
    pub results: SmallVec<[Val; 1]>, 
    pub attributes: AttributeMap,
    pub regions: SmallVec<[RegionId; 1]>,
    pub custom_data: OpStorage,
}

pub struct OpStorage {
    data: SmallVec<[u8; 32]>,
    drop_fn: Option<fn(&mut SmallVec<[u8; 32]>)>,
}

impl OpStorage {
    pub fn new() -> Self {
        Self {
            data: SmallVec::new(),
            drop_fn: None,
        }
    }

    pub fn from_value<T: 'static>(value: T) -> Self {
        let mut storage = Self::new();
        storage.write(&value);
        std::mem::forget(value);
        storage
    }

    pub fn write<T: 'static>(&mut self, value: &T) {
        self.data.clear();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                value as *const T as *const u8,
                std::mem::size_of::<T>(),
            )
        };
        self.data.extend_from_slice(bytes);
        
        if std::mem::needs_drop::<T>() {
            self.drop_fn = Some(|data| {
                unsafe {
                    std::ptr::drop_in_place(data.as_mut_ptr() as *mut T);
                }
            });
        }
    }

    pub fn as_ref<T: 'static>(&self) -> Option<&T> {
        if self.data.len() == std::mem::size_of::<T>() {
            Some(unsafe { &*(self.data.as_ptr() as *const T) })
        } else {
            None
        }
    }
}

impl Drop for OpStorage {
    fn drop(&mut self) {
        if let Some(drop_fn) = self.drop_fn {
            drop_fn(&mut self.data);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Value {
    pub name: Option<StringId>,
    pub ty: TypeId,
    pub defining_op: Option<OpRef>,
}

pub struct OpRegistry {
    ops: ahash::AHashMap<(StringId, StringId), &'static OpInfo>,
}

impl OpRegistry {
    pub fn new() -> Self {
        Self {
            ops: ahash::AHashMap::new(),
        }
    }

    pub fn register(&mut self, dialect: StringId, name: StringId, info: &'static OpInfo) {
        self.ops.insert((dialect, name), info);
    }

    pub fn get(&self, dialect: StringId, name: StringId) -> Option<&'static OpInfo> {
        self.ops.get(&(dialect, name)).copied()
    }

    pub fn register_builtin_ops(&mut self, string_interner: &mut crate::string_interner::StringInterner) {
        // Register all operations collected by inventory
        for info in inventory::iter::<&'static OpInfo> {
            let dialect = string_interner.intern(info.dialect);
            let name = string_interner.intern(info.name);
            self.register(dialect, name, *info);
        }
        
        // Also register manually defined ops
        use crate::dialects::arith::{ConstantOp, AddOp, MulOp};
        
        for info in &[ConstantOp::INFO, AddOp::INFO, MulOp::INFO] {
            let dialect = string_interner.intern(info.dialect);
            let name = string_interner.intern(info.name);
            self.register(dialect, name, *info);
        }
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Op: Sized + 'static {
    fn info(&self) -> &'static OpInfo;
}

// Removed register_op! macro since we're not using inventory anymore