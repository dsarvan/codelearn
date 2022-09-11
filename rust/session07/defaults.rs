use std::default::Default;

#[derive(Debug)]
struct D {
    x: i32,
    y: i32,
    z: i32,
}

impl D {
    fn new(x: i32, y: i32, z: Option<i32>) -> Self { /* constructor */
        match z {
            Some(z) => Self { x, y, z }, /* if user provide z value */
            None => Self { x, y, z: 0 }, /* if user not provide z value */
        }
    }
}

impl Default for D {
    fn default() -> Self {
        Self { x: 0, y: 0, z: 0 }
    }
}

fn main() {
    let d1 = D::new(1, 2, None);
    let d2 = D {
        x: 1,
        y: 2,
        ..D::default()
    };
    println!("{:?}", d1);
    println!("{:?}", d2);
}
