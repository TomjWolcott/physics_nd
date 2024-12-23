use nalgebra::{Matrix, SMatrix, SVector, Vector};
use wedged::algebra::*;
use wedged::subspace::*;
use wedged::base::{AllocBlade, AllocMultivector, Const, Inv, U2, Zero};

pub const N: usize = 4;
pub const K: usize = N*(N-1)/2;
pub const K_SQUARED: usize = K*K;
pub type FLOAT = f64;

/// Extracts the indices for a bivector in wedged
fn bivector_indices() -> [(usize, usize); K] {
    let mut indices = [(0, 0); K];

    for i in 0..K {
        for j in 0..N {
            let b: VecN<_, _> = BiVecN::<FLOAT, Const<N>>::basis(i) % VecN::<FLOAT, Const<N>>::basis(j);

            if !b.is_zero() {
                let j2 = b.iter().position(|x| *x != 0.0).unwrap();
                indices[i] = (j, j2);
                break;
            }
        }
    }

    indices
}

/// Used to generate the commutator matrix code, this could be a proc macro,
/// but I don't feel like going through the hassle, so just run this test and copy-paste
/// the result
#[test]
fn generate_commutator_matrix() {
    const PARAM_NAME: &'static str = "l";
    const TABS: &'static str = "    ";

    let n = 4;
    let k = n * (n - 1) / 2;

    let commutator_matrix = operator_matrix(
        commutator,
        &BiVec4::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    );

    let mut code = String::from("SMatrix::<FLOAT, K, K>::new(");

    for i in 0..k {
        code.push('\n');
        code.push_str(TABS);

        for j in 0..k {
            code = match commutator_matrix[(i, j)] {
                0.0 => format!("{}  {}0.0,", code, " ".repeat(PARAM_NAME.len())),
                x @ ..0.0 => format!("{} -{}[{}],", code, PARAM_NAME, x.abs() as usize - 1),
                x@ 0.0.. => format!("{}  {}[{}],", code, PARAM_NAME, x as usize - 1),
                x => panic!("Bad element in commutator matrix: {x}")
            }
        }
    }

    println!("{code}\n)")
}

pub fn operator_matrix(
    func: impl Fn(&BiVec4<FLOAT>, &BiVec4<FLOAT>) -> BiVec4<FLOAT>,
    b: &BiVec4<FLOAT>
) -> SMatrix<FLOAT, K, K> {
    let mut matrix: SMatrix<FLOAT, K, K> = SMatrix::zeros();

    for i in 0..K {
        let mut rhs = BiVec4::zero();
        rhs[i] = 1.0;

        let result = func(b, &rhs);

        matrix.set_column(i, &result.as_vector());
    }

    matrix
}

/// Computes the matrix \[ω\]_x such that \[ω\]_x * ν = ω ⨯ ν = (ων - νω)/2
pub fn commutator_matrix(l: &BiVecN<FLOAT, Const<N>>) -> SMatrix<FLOAT, K, K> {
    SMatrix::<FLOAT, K, K>::new(
        0.0,  l[2], -l[1],   0.0,  l[5], -l[4],
        -l[2],   0.0,  l[0], -l[5],   0.0,  l[3],
        l[1], -l[0],   0.0,  l[4], -l[3],   0.0,
        0.0,  l[5], -l[4],   0.0,  l[2], -l[1],
        -l[5],   0.0,  l[3], -l[2],   0.0,  l[0],
        l[4], -l[3],   0.0,  l[1], -l[0],   0.0,
    )
}

#[derive(Debug)]
pub struct Pose {
    pub pos: Vec4<FLOAT>,
    pub ori: Rotor<FLOAT, Const<N>>
}

impl Default for Pose {
    fn default() -> Self {
        Pose {
            pos: Vec4::zero(),
            ori: Rotor::one_generic(Const::<N>)
        }
    }
}

#[derive(Debug)]
pub struct Rate {
    pub linear: Vec4<FLOAT>,
    pub angular: BiVecN<FLOAT, Const<N>>
}

impl Default for Rate {
    fn default() -> Self {
        Rate {
            linear: Vec4::zero(),
            angular: BiVecN::zero()
        }
    }
}

// pub struct Acceleration {
//     pub linear: Vec4<FLOAT>,
//     pub angular: BiVecN<FLOAT, Const<N>>
// }

#[derive(Debug)]
pub struct Momentum {
    pub linear: Vec4<FLOAT>,
    pub angular: BiVecN<FLOAT, Const<N>>
}

#[derive(Debug)]
pub struct Forque {
    pub linear: Vec4<FLOAT>,
    pub angular: BiVecN<FLOAT, Const<N>>
}

#[derive(Debug)]
pub struct Inertia {
    pub mass: FLOAT,
    pub inertia_tensor: SMatrix<FLOAT, K, K>,
    pub inv_inertia_tensor: SMatrix<FLOAT, K, K>,
}

impl Inertia {
    pub fn from_diagonal(diagonal: [FLOAT; K], mass: FLOAT) -> Self {
        let mut inertia_tensor = SMatrix::zeros();
        let mut inv_inertia_tensor = SMatrix::zeros();

        for i in 0..K {
            inertia_tensor[(i, i)] = diagonal[i];
            inv_inertia_tensor[(i, i)] = diagonal[i].inv();
        }

        Self {
            mass,
            inertia_tensor,
            inv_inertia_tensor
        }
    }

    pub fn cuboid(size: Vec4<FLOAT>, mass: FLOAT) -> Self {
        let mut diagonal = [0.0; K];

        for (i, (i1, i2)) in bivector_indices().iter().enumerate() {
            diagonal[i] = mass / 12.0 * (size[*i1] * size[*i1] + size[*i2] * size[*i2]);
        }

        Self::from_diagonal(diagonal, mass)
    }

    pub fn n_sphere(radius: FLOAT, mass: FLOAT) -> Self {
        Self::from_diagonal([2.0 / 5.0 * mass * radius * radius; K], mass)
    }
}

pub trait IntoNAlgebraVector<const D: usize>  {
    fn as_vector(&self) -> SVector<FLOAT, D>;
}

// for bivec
impl IntoNAlgebraVector<K> for BiVecN<FLOAT, Const<N>> {
    fn as_vector(&self) -> SVector<FLOAT, K> {
        let mut vector = SVector::zeros();
        for i in 0..K {
            vector[i] = self[i];
        }

        vector
    }
}

pub trait IntoWedgedBlade<const G: usize> where FLOAT: AllocBlade<Const<N>, Const<G>> {
    fn as_blade(&self) -> Blade<FLOAT, Const<N>, Const<G>>;
}

// for into grade2
impl IntoWedgedBlade<2> for SVector<FLOAT, K> {
    fn as_blade(&self) -> Blade<FLOAT, Const<N>, Const<2>> {
        let mut blade = Blade::zero();
        for i in 0..K {
            blade[i] = self[i];
        }

        blade
    }
}

pub fn map_bivector(inertia_matrix: &SMatrix<FLOAT, K, K>, bivector: &BiVecN<FLOAT, Const<N>>) -> BiVecN<FLOAT, Const<N>> {
    let mut result = BiVecN::zero();

    for i in 0..K {
        for j in 0..K {
            result[i] += inertia_matrix[(i, j)] * bivector[j]
        }
    }

    result
}

pub fn commutator<const DIM: usize>(
    b1: &BiVecN<FLOAT, Const<DIM>>,
    b2: &BiVecN<FLOAT, Const<DIM>>
) -> BiVecN<FLOAT, Const<DIM>> where
    FLOAT: AllocBlade<Const<DIM>, Const<2>> + AllocMultivector<Const<DIM>>
{
    (b1 * b2 - b2 * b1).select_grade::<Const<2>>() / 2.0
}