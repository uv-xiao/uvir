use crate::string_interner::{StringInterner, StringId};
use crate::types::{TypeInterner, TypeId, TypeKind};
use crate::ops::OpRegistry;
use crate::region::{RegionManager, RegionId};

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
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}