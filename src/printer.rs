use std::fmt::Write;
use crate::error::Result;

pub struct Printer {
    output: String,
    indent_level: usize,
    indent_str: String,
}

impl Printer {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent_level: 0,
            indent_str: "  ".to_string(),
        }
    }

    pub fn print(&mut self, s: &str) -> Result<()> {
        self.output.push_str(s);
        Ok(())
    }

    pub fn println(&mut self, s: &str) -> Result<()> {
        writeln!(&mut self.output, "{}", s)
            .map_err(|_| crate::error::Error::InternalError("Write error".to_string()))
    }

    pub fn print_indent(&mut self) -> Result<()> {
        for _ in 0..self.indent_level {
            self.output.push_str(&self.indent_str);
        }
        Ok(())
    }

    pub fn indent(&mut self) {
        self.indent_level += 1;
    }

    pub fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    pub fn get_output(self) -> String {
        self.output
    }

    pub fn clear(&mut self) {
        self.output.clear();
        self.indent_level = 0;
    }
}

impl Default for Printer {
    fn default() -> Self {
        Self::new()
    }
}