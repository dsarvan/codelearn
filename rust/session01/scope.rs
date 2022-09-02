fn main() {
    // scope
    // This binding lives in the main function
    let long_lived_binding = 1;

    // In Rust, a code block is also a scope block and at the end of the
    // scope it will clean up memory for you therefore you could limit the
    // resource usage with this for example.
    // This is a block, and has a smaller scope than the main function
    {
        // This binding only exists in this block
        let short_lived_binding = 2;

        println!("inner short: {}", short_lived_binding);

        // This binding *shadows* the outer one
        let long_lived_binding = 5_f32;

        println!("inner long: {}", long_lived_binding);
    }
    // End of the block

    // Error! 'short_lived_binding' doesn't exist in this scope
    //println!("outer short: {}", short_lived_binding);

    println!("outer long: {}", long_lived_binding);

    // This binding also *shadows* the previous binding
    let long_lived_binding = 'a';

    println!("outer long: {}", long_lived_binding);
}
