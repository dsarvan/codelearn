fn main() {
    let x = 1;
    println!("x: {}", x);
    // binds x again, shadowing the old one from above
    let x = 'i';
    println!("x: {}", x);

    // declare, initialize
    let something; // declared but not initialized
    let x = 5;
    //println!("x, something: {}, {}", x, something);
    something = x * 5;
    println!("x, something: {}, {}", x, something);

    // mutability
    let mut y = 0;
    y = y * 2 + x;
    dbg!(y); // print and informs where the code is located

    const BLAH: i32 = 42; // constant should be uppercase
    y *= BLAH;
    dbg!(y);
}
