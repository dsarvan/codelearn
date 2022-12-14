//! example showing uppercasing for unicode and ASCII

fn naive_capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

fn ascii_capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(), // if c.next == None then return an empty stirng
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

fn main() {
    let names: Vec<String> = vec![
        "alice".to_string(),
        "bob".to_string(),
        "ozgul".to_string(),
        "gad".to_string(),
    ];
    for name in names.iter() {
        println!("{} -> {}", name, naive_capitalize(name));
        println!("{} -> {}", name, ascii_capitalize(name));
        println!("");
    }
}
