pub mod components;

use nalgebra::SMatrix;
use wedged::algebra::{BiVec4, Vec4};
use wedged::base::{Const, Inv, Zero};
use wedged::subspace::{Rotor, Rotor4};
use components::*;

// References:
// Marc Ten Bosch 2020: "𝑁-Dimensional Rigid Body Dynamics", https://marctenbosch.com/ndphysics/NDrigidbody.pdf
// Christian Perwass 2009: "Geometric Algebra with Applications in Engineering", https://link.springer.com/book/10.1007/978-3-540-89068-3
// Erin Catto 2015: "Physics for Game Programme", https://www.gdcvault.com/play/1022197/Physics-for-Game-Programmers-Numerical

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
    rate.angular += dt * pose.ori.inv().rot(map_bivector(&inertia.inv_inertia_tensor, &pose.ori.rot(&forque.angular)));

    // R += (-0.5 * w * R) * dt
    let dR_dt = -0.5 * dt * &rate.angular * pose.ori.into_even();
    pose.ori = (pose.ori.into_even() - dR_dt.select_even()).into_rotor_unchecked();

    pose.ori = reconstruct_versor(pose.ori.into_versor_unchecked())
        .try_into_even()
        .unwrap()
        .into_rotor_unchecked();
}

/// (Adapted from slide 76 in Catto 2015) Solves for the gyroscopic term of angular velocity update using a single Newton-Raphson iteration
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

    // println!("\
    //     Solve Gyroscopic Term:\
    //     \n  f = {:.5}\
    //     \n  ang_vel = {:.5}\
    //     \n  body_ang_vel = {:.5}\
    //     \n  new_body_ang_vel = {:.5}\
    //     \n  new_ang_vel = {:.5}\
    //     \n  jacobian = N/A", f, ang_vel, body_ang_vel, new_body_ang_vel, new_ang_vel);

    new_ang_vel
}

pub fn proj_sphere(radius: FLOAT) -> (impl Fn(&Vec4<FLOAT>) -> (Vec4<FLOAT>, Vec4<FLOAT>)) {
    move |pos| (radius * pos.normalize(), pos.normalize())
}

pub fn proj_torus(r1: FLOAT, r2: FLOAT) -> (impl Fn(&Vec4<FLOAT>) -> (Vec4<FLOAT>, Vec4<FLOAT>)) {
    move |pos| {
        let xz_pos = r1 * Vec4::new(pos[0], 0.0, pos[2], 0.0).normalize();

        let normal = (pos - xz_pos).normalize();

        (xz_pos + r2 * normal, normal)
    }
}

pub fn project_onto_surface(mut pose: &mut Pose, mut rate: &mut Rate, proj: impl Fn(&Vec4<FLOAT>) -> (Vec4<FLOAT>, Vec4<FLOAT>)) {
    let Pose { pos, ori } = &mut pose;
    let (new_pos, normal) = proj(&*pos);

    *pos = new_pos;
    let rate_norm = rate.linear.norm();
    rate.linear -= ((rate.linear % normal) * normal).select_grade::<Const<1>>();
    // rate.linear = rate.linear.normalize() * rate_norm;

    let ori_normal_dir: Vec4<FLOAT> = ori.rot(Vec4::new(1.0, 0.0, 0.0, 0.0));
    let mid_vector = (ori_normal_dir - normal).normalize();
    let correction_rotor = (mid_vector * ori_normal_dir).select_even().into_rotor_unchecked();

    *ori = (*correction_rotor * **ori).into_rotor_unchecked();
    // investigate rot stuff
    // println!("proj angular stuff:\n  ori: {:.4}\n  ori_normal_dir: {:.4}\n  mid_vector: {:.4}\n  correction_rotor: {:.4}\n  new_ori_rot_dir: {:.4}\n  normal: {normal:.4}\n  pos: {pos:.4}\n\n", ori, ori_normal_dir, mid_vector, correction_rotor, ori.rot(Vec4::new(1.0, 0.0, 0.0, 0.0)));

    rate.angular -= ((rate.angular % normal) * normal).select_grade::<Const<2>>();

    // println!("rate_norm: {rate_norm:.5}")
}

pub fn cuboid_solve_wall_collision(
    wall_pos: Vec4<FLOAT>,
    wall_normal: Vec4<FLOAT>,
    pose: &mut Pose,
    rate: &mut Rate,
    inertia: &Inertia,
    restitution: FLOAT,
    cuboid_size: Vec4<FLOAT>
) {
    let Pose { pos, ori } = pose;
    let mut contact_point = None;

    for i in 0..16 {
        let corner: Vec4<FLOAT> = *pos + ori.rot(Vec4::new(
            (2.0 * (i & 1) as f64 - 1.0) * cuboid_size[0] / 2.0,
            (1.0 * (i & 2) as f64 - 1.0) * cuboid_size[1] / 2.0,
            (0.5 * (i & 4) as f64 - 1.0) * cuboid_size[2] / 2.0,
            (0.25 * (i & 8) as f64 - 1.0) * cuboid_size[3] / 2.0,
        ));

        if ((corner - wall_pos) % wall_normal)[0] < 0.0 {
            contact_point = if let Some((i, point, normal)) = contact_point {
                Some((i + 1.0, ((corner + i * point) / (i + 1.0)), normal))
            } else {
                Some((1.0, corner, wall_normal))
            };
        }
    }

    if let Some((i, point, normal)) = contact_point {
        println!("CONTACT\n  point: {point:.5} (#points={i})\n  normal: {normal:.5}");
        let contact_arm = point - *pos;
        let angular_plane = pose.ori.inv().rot(contact_arm ^ normal);
        let effective_mass = 1.0 / (inertia.mass - (angular_plane % map_bivector(&inertia.inv_inertia_tensor, &angular_plane))[0]);
        let vel_constraint = (rate.linear % normal - angular_plane % pose.ori.inv().rot(rate.angular))[0];
        let impulse_magnitude = -effective_mass * vel_constraint;
        rate.linear += impulse_magnitude * normal / inertia.mass;
        rate.angular += impulse_magnitude * pose.ori.rot(map_bivector(&inertia.inv_inertia_tensor, &angular_plane))
    }
}