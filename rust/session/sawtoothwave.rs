// File: sawtoothwave.rs
// Name: D.Saravanan
// Date: 26/08/2022

/* Sawtooth wave by summing sine wave */

use ndarray::prelude::*;
use std::f64::consts::PI;

fn main() {
    let xval = Array::linspace(-2.*PI, 2.*PI, 5000);
    let sawtooth_wave = xval.mapv(f64::sin);

    for n in range(2..10001).step_by(1) {
        sawtooth_wave += (-1)**(n-1) * (1/n) * (n*xval).mapv(f64::sin);
    }
}
