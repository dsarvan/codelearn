fn main() {
    // integers
    let x: i32 = -1000;
    let x = -1000i32;
    let y: u64 = 100; // unsigned 64 bit integer
    let y = 100u64;
    let large_number = 123_456_789;

    // floats
    let xf: f32 = -1.2345;
    let xf = -1.2345_f32;

    // complex numbers
    // not part of rust standard library
    let complex_integer = num::complex::Complex::new(10, 20);
    let complex_float = num::complex::Complex::new(10.0, 20.1);

    // binary literals
    let x: u8 = 0b1010_1010;
    let y: u8 = 0b0101_0101;
    dbg!(x | y);

    // hex
    let x: u8 = 0xAF;
    let largex = 0xABCD_EF01_u32;

    // boolean
    let yes = true;
    let no = false;
}
