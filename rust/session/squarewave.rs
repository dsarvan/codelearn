// File: squarewave.rs
// Name: D.Saravanan
// Date: 26/08/2022

/* Square wave by summing sine wave */

use ndarray::prelude::*;
use std::f64::consts::PI;

fn main() {
    let xval = Array::linspace(-2.*PI, 2.*PI, 5000);
    let square_wave = xval.mapv(f64::sin);

    for n in (3..10001).step_by(2) {
        square_wave += (1/n) * (n*xval).mapv(f64::sin);
    }
}
