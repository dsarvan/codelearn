fn main() {
    let alice = String::from("I like dogs"); // construct a string from("input")
    let bob: String = alice.replace("dog", "cat");

    println!("Alice says: {}", alice);
    println!("Bob says: {}", bob);
    println!("Bob says again: {}", bob);

    let pangram: &'static str = "the quick brown fox jumps over the lazy";
    println!("Pangram: {}", pangram);

    println!("Words in reverse");
    for word in pangram.split_whitespace().rev() {
        println!("> {}", word);
    }

    let mut chars: Vec<char> = pangram.chars().collect();
    chars.sort();       // sort
    chars.dedup();      // deduplicate (unique)

    let mut string = String::new();
    for c in chars {
        string.push(c);
        string.push_str(", ");
    }

    let chars_to_trim: &[char] = &[' ', ','];
    let trimmed_str: &str = string.trim_matches(chars_to_trim);
    println!("Used    characters: {}", string);
    println!("Trimmed characters: {}", trimmed_str);
}
