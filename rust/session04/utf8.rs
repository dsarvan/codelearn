fn main() {
    let ascii = String::from("Hello");
    // string storage -> memory, how long is it, capacity of the string
    // in 32-bit machine: (3 x 4 bytes) + len of the string = 12 + 5 = 17 bytes
    println!(
        "'{}': lenght: {}, chars: {}, mem_size: {} bytes",
        ascii,
        ascii.len(),
        ascii.chars().count(),
        std::mem::size_of::<String>() + ascii.len()
    );


    println!("{:?}", ascii.as_bytes()); // unsigned 8-bit integers
    println!("after e: {:?}", &ascii[2..]);

    println!("---");

    let uni = String::from("Hello");
    println!(
        "'{}': length: {}, chars: {}, mem_size: {} bytes",
        uni,
        uni.len(),
        uni.chars().count(),
        std::mem::size_of::<String>() + uni.len()
    );

    println!("{:?}", uni.as_bytes());
    println!("after e: {:?}", &uni[2..]);
}
