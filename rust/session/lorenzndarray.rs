// File: lorenzndarray.rs
// Name: D.Saravanan
// Date: 20/09/2022

/* Lorenz system */

use ndarray::prelude::*;

fn main() {
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
}
