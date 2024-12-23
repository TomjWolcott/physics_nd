pub mod components;

use nalgebra::SMatrix;
use wedged::algebra::{BiVec4, Vec4};
use wedged::base::{Const, Inv, Zero};
use wedged::subspace::{Rotor, Rotor4};
use components::*;

// 4D physics using:
// https://marctenbosch.com/ndphysics/NDrigidbody.pdf
// https://www.gdcvault.com/play/1022197/Physics-for-Game-Programmers-Numerical


#[test]
fn test_commutator_stuff() {
    let l = BiVec4::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    let l_comm = operator_matrix(commutator, &l);
    let v = BiVec4::new(5.0, -2.0, 1.0, -3.0, 3.0, 1.0);

    println!("l: {}\ncommutator matrix: {}", l, l_comm);

    println!("l_comm * v = {}", (l_comm * v.as_vector()).as_blade());
    println!("     l x v = {}", commutator(&l, &v));
}

#[test]
fn test_comm_deriv_stuff() {
    let inertia = Inertia::cuboid(Vec4::new(10.0, 59.0, 0.22, 5.0), 3.3);
    let ang_vel = BiVec4::new(1.0, 10.0, -5.0, -1.0, 58.0, 6.0);
    let mut deriv1: SMatrix<FLOAT, K, K> = SMatrix::zero();

    for i in 0..K {
        let mut b = BiVec4::zero();
        b[i] = 1.0;

        deriv1.set_column(
            i,
            &(operator_matrix(commutator, &b) * inertia.inertia_tensor * ang_vel.as_vector())
        );
    }

    let deriv2 = -operator_matrix(
        commutator,
        &(inertia.inertia_tensor * ang_vel.as_vector()).as_blade()
    );

    println!("{}\n{}", deriv1, deriv2);
    assert_eq!(deriv1, deriv2);
}


#[test]
fn gravity_and_const_torque() {
    let mut pose = Pose::default();
    let mut rate = Rate::default();
    let inertia = Inertia::cuboid(Vec4::new(1.0, 1.0, 1.0, 1.0), 1.0);
    let forque = Forque {
        linear: Vec4::new(0.0, 0.0, 0.0, 0.0),
        angular: BiVec4::new(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
    };

    let dt = 0.01;
    let steps = 10;
    let substeps = 100;

    for i in 0..steps {
        for _ in 0..substeps {
            integrate(&mut pose, &mut rate, &inertia, &forque, dt);
        }

        println!("t = {}s, pose: {:?}, rate: {:?}", (i * substeps) as FLOAT * dt, pose, rate);
    }


}

/// Integrate the pose and rate of a rigid body using semi-implicit Euler integration
pub fn integrate(pose: &mut Pose, rate: &mut Rate, inertia: &Inertia, forque: &Forque, dt: FLOAT) {
    // Semi-implicit Euler integration
    rate.linear += forque.linear * dt / inertia.mass;
    pose.pos += rate.linear * dt;

    rate.angular = solve_gyroscopic_term(&pose.ori, &inertia, &rate.angular, dt);
    // ω_2 = ω_1 + dt * ŘI[RτŘ]R (convert torque to body frame, apply inertia, convert back to world frame)
    println!("  rate.angular: {}", rate.angular);
    rate.angular += dt * pose.ori.inv().rot(map_bivector(&inertia.inv_inertia_tensor, &pose.ori.rot(&forque.angular)));
    println!("  post-torque rate.angular: {}", rate.angular);

    // R += (-0.5 * w * R) * dt
    let dR_dt = -0.5 * dt * &rate.angular * pose.ori.into_even();
    pose.ori = (pose.ori.into_even() - dR_dt.select_even()).into_rotor_unchecked();

    pose.ori = pose.ori.normalize().into_rotor_unchecked();
}

/// Solves for the gyroscopic term of angular velocity update using a single Newton-Raphson iteration
fn solve_gyroscopic_term(ori: &Rotor4<FLOAT>, inertia: &Inertia, ang_vel: &BiVec4<FLOAT>, dt: FLOAT) -> BiVec4<FLOAT> {
    let body_ang_vel = ori.inv().rot(ang_vel);
    let body_ang_momentum = map_bivector(&inertia.inertia_tensor, &body_ang_vel);

    let f = dt * commutator(&body_ang_vel, &body_ang_momentum);
    let jacobian = inertia.inertia_tensor + dt * (
        commutator_matrix(&body_ang_vel) * inertia.inertia_tensor - commutator_matrix(&body_ang_momentum)
    );

    let new_body_ang_vel = body_ang_vel - jacobian.lu()
        .solve(&f.as_vector())
        .expect(format!(
            "Failed to solve for body angular velocity.  f = {:.5}, ω_1 = {:.5}, L_1 = {:.5}, jacobian = {:.5}",
            f, body_ang_vel, body_ang_momentum, jacobian
        ).as_str()).as_blade();

    let new_ang_vel = ori.rot(&new_body_ang_vel);

    println!("\
        Solve Gyroscopic Term:\
        \n  f = {:.5}\
        \n  ang_vel = {:.5}\
        \n  body_ang_vel = {:.5}\
        \n  new_body_ang_vel = {:.5}\
        \n  new_ang_vel = {:.5}\
        \n  jacobian = N/A", f, ang_vel, body_ang_vel, new_body_ang_vel, new_ang_vel);

    new_ang_vel
}