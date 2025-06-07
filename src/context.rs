use crate::string_interner::{StringInterner, StringId};
use crate::types::{TypeInterner, TypeId, TypeKind, FloatPrecision};
use crate::ops::{OpRegistry, OpData, Val};
use crate::region::{RegionManager, RegionId};
use crate::error::Result;

pub struct Context {
    pub strings: StringInterner,
    pub types: TypeInterner,
    pub ops: OpRegistry,
    pub regions: RegionManager,
    pub global_region: RegionId,
}

impl Context {
    pub fn new() -> Self {
        let mut regions = RegionManager::new();
        let global_region = regions.create_region();
        
        let mut ctx = Self {
            strings: StringInterner::new(),
            types: TypeInterner::new(),
            ops: OpRegistry::new(),
            regions,
            global_region,
        };

        ctx.ops.register_builtin_ops(&mut ctx.strings);
        ctx
    }

    pub fn intern_string(&mut self, s: &str) -> StringId {
        self.strings.intern(s)
    }

    pub fn get_string(&self, id: StringId) -> Option<&str> {
        self.strings.get(id)
    }

    pub fn intern_type(&mut self, kind: TypeKind) -> TypeId {
        self.types.intern(kind)
    }

    pub fn get_type(&self, id: TypeId) -> Option<&TypeKind> {
        self.types.get(id)
    }

    pub fn create_region(&mut self) -> RegionId {
        self.regions.create_region()
    }

    pub fn get_region(&self, id: RegionId) -> Option<&crate::region::Region> {
        self.regions.get_region(id)
    }

    pub fn get_region_mut(&mut self, id: RegionId) -> Option<&mut crate::region::Region> {
        self.regions.get_region_mut(id)
    }

    pub fn get_global_region(&self) -> &crate::region::Region {
        self.regions.get_region(self.global_region)
            .expect("Global region should always exist")
    }

    pub fn get_global_region_mut(&mut self) -> &mut crate::region::Region {
        self.regions.get_region_mut(self.global_region)
            .expect("Global region should always exist")
    }

    // Return the global region ID
    pub fn global_region(&self) -> RegionId {
        self.global_region
    }

    // Create a new value in a region
    pub fn create_value(&mut self, name: Option<&str>, ty: TypeId) -> Val {
        let name_id = name.map(|n| self.intern_string(n));
        let region = self.get_global_region_mut();
        region.create_value(name_id, ty)
    }

    // Get the operation registry
    pub fn op_registry(&self) -> &OpRegistry {
        &self.ops
    }

    // Set the type of a value
    pub fn set_value_type(&mut self, val: Val, ty: TypeId) {
        // Find the value in all regions and update its type
        for region in self.regions.regions.values_mut() {
            if let Some(value) = region.values.get_mut(val) {
                value.ty = ty;
                return;
            }
        }
    }

    // Add an operation to a region
    pub fn add_operation(&mut self, region_id: RegionId, op_data: OpData) -> Result<()> {
        if let Some(region) = self.regions.get_region_mut(region_id) {
            region.add_operation(op_data);
            Ok(())
        } else {
            Err(crate::error::Error::InvalidRegion(format!("Region {:?} not found", region_id)))
        }
    }

    // Get builtin types helper
    pub fn builtin_types(&mut self) -> BuiltinTypes {
        BuiltinTypes { ctx: self }
    }
}

// Helper struct for builtin types
pub struct BuiltinTypes<'a> {
    ctx: &'a mut Context,
}

impl<'a> BuiltinTypes<'a> {
    pub fn i1(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 1, signed: false })
    }

    pub fn i8(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 8, signed: true })
    }

    pub fn i16(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 16, signed: true })
    }

    pub fn i32(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 32, signed: true })
    }

    pub fn i64(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 64, signed: true })
    }

    pub fn u8(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 8, signed: false })
    }

    pub fn u16(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 16, signed: false })
    }

    pub fn u32(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 32, signed: false })
    }

    pub fn u64(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Integer { width: 64, signed: false })
    }

    pub fn f16(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Half })
    }

    pub fn f32(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Single })
    }

    pub fn f64(&mut self) -> TypeId {
        self.ctx.intern_type(TypeKind::Float { precision: FloatPrecision::Double })
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}