use eyre::{eyre, Result};
use ndarray::prelude::*;
use gnuplot::{Figure, AxesCommon, Caption, Color};
use std::f64::consts::PI;

fn double_first(vec: Vec<&str>) -> Result<i32> {
    let first = vec.first().ok_or_else(|| eyre!("no first item"))?;
    let parsed = first.parse::<i32>()?;
    Ok(2 * parsed)
}

fn print(result: Result<i32>) {
    match result {
        Ok(n) => println!("The first doubled is {}", n),
        Err(e) => println!("Error: {}", e),
    }
}

fn main() {
    let numbers = vec!["42", "93", "18"];
    let empty = vec![];
    let strings = vec!["tofu", "93", "18"];

    print(double_first(numbers));
    print(double_first(empty));
    print(double_first(strings));

    /* lorenz system */
    const N: usize = 5000;
    let mut x = Array::<f64, _>::ones(N);
    let mut y = Array::<f64, _>::ones(N);
    let mut z = Array::<f64, _>::ones(N);

    let delt = 0.01;
    let (sigma, beta, rho) = (10.0, 8.0 / 3.0, 28.0);

    for n in 0..N - 1 {
        x[n + 1] = x[n] + delt * (sigma * (y[n] - x[n]));
        y[n + 1] = y[n] + delt * (x[n] * (rho - z[n]) - y[n]);
        z[n + 1] = z[n] + delt * (x[n] * y[n] - beta * z[n]);
    }

    println!("{:?}", x);
    println!("{:?}", y);
    println!("{:?}", z);

//    let mut fg = Figure::new();
//    fg.axes2d()
//        .lines(&x, &z, &[Caption("x - z plot"), Color("red")])
//        .set_title("Lorenz system", &[])
//        .set_x_label("x", &[]).set_y_label("z", &[])
//        .set_x_grid(true).set_y_grid(true);
//    fg.show().unwrap();

    /* curves for the mathematically curious */
    let x1 = Array::linspace(-2.*PI, 2.*PI, N);
    let x2 = Array::linspace(-2.*PI, 2.*PI, N);

    let val1 = x1.mapv(f64::sin) + x2.mapv(f64::cos) ;
    let val2 = x1.mapv(f64::sin) * x2.mapv(f64::sin) + x1.mapv(f64::cos);
    let curve = val1.mapv(f64::sin) - val2.mapv(f64::cos);

    let mut xy = Figure::new();
    xy.axes2d()
        .lines(&x, &curve, &[Caption("sin(sin x + cos y) = cos(sin xy + cos x)"), Color("black")])
        .set_title("Curves for the Mathematically Curious", &[])
        .set_x_grid(true).set_y_grid(true);
    xy.show().unwrap();
}
