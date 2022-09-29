use ndarray::prelude::*;

fn main() {
    // This is an example where the compiler can infer the element type
    // because `f64::sin` can only be called on `f64` elements:
    let arr1 = Array::zeros((3, 2, 4).f());
    println!("{}", arr1.mapv(f64::sin));

    // Specify just the element type and infer the dimensionality:
    let arr2 = Array::<f64, _>::zeros((3, 2, 4).f());
    let arr3: Array<f64, _> = Array::zeros((3, 2, 4).f());

    // Specify both the element type and dimensionality:
    let arr4 = Array3::<f64>::zeros((3, 2, 4).f())
    let arr5: Array3<f64> = Array::zeros((3, 2, 4).f());
    let arr6 = Array::<f64, Ix3>::zeros((3, 2, 4).f());
    let arr7: Array<f64, Ix3> = Array::zeros((3, 2, 4).f());
}
