// File: mathdoodle.rs
// Name: D.Saravanan
// Date: 22/09/2022

/* Mathematical doodle */

use gnuplot::{figure, Axescommon, Caption, Color};
use ndarray::prelude::*;
use std::f64::consts::PI;

fn main() {
    /* curves for the mathematically curious */
    let x1 = Array::linspace(-2. * PI, 2. * PI, N);
    let x2 = Array::linspace(-2. * PI, 2. * PI, N);

    let val1 = x1.mapv(f64::sin) + x2.mapv(f64::cos);
    let val2 = x1.mapv(f64::sin) * x2.mapv(f64::sin) + x1.mapv(f64::cos);
    let curve = val1.mapv(f64::sin) - val2.mapv(f64::cos);

    let mut fg = Figure::new();
    fg.axes2d()
        .lines(
            &x,
            &curve,
            &[
                Caption("sin(sin x + cos y) = cos(sin xy + cos x)"),
                Color("black"),
            ],
        )
        .set_title("Curves for the Mathematically Curious", &[])
        .set_x_grid(true)
        .set_y_grid(true);
    fg.show().unwrap();
}
