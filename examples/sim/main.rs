//! A simple 3D scene with light shining over a cube sitting on a plane.

use std::any::{type_name, type_name_of_val};
use std::f32::consts::PI;
use bevy::color::palettes::basic::GREEN;
use bevy::color::palettes::css::{BLACK, RED, WHITE};
use bevy::pbr::CascadeShadowConfigBuilder;
use bevy::prelude::*;
use bevy::render::camera::ScalingMode;
use physics_nd::components::*;
use physics_nd::*;
use wedged::algebra as ga;
use wedged::base::{Const, Zero};

const DT: f64 = 0.0001;
const N_SUBSTEPS: usize = 1;

fn main() {
    println!("Hello, world!");

    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(BLACK.into()))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, physics_update)
        .run();
}

#[derive(Component)]
struct Object {
    pose: Pose,
    rate: Rate,
    inertia: Inertia,
    forque: Forque
}

const LENGTHS: [f64; 3] = [0.3, 2.0, 4.0];

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let angle = (0.0_f64).to_radians() / 2.0;

    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(LENGTHS[0] as f32, LENGTHS[1] as f32, LENGTHS[2] as f32))),
        MeshMaterial3d(materials.add(Color::srgb_u8(244, 234, 155))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Object {
            pose: Pose {
                pos: ga::Vec4::new(0.0, -0.0, 0.0, 0.0),
                ori: (ga::Vec4::new(0.0, 1.0, 0.0, 0.0) * ga::Vec4::new(0.0, angle.cos(), angle.sin(), 0.0)).select_even().into_rotor_unchecked()
            },
            rate: Rate {
                linear: ga::Vec4::new(0.0, 0.0, 0.0, 0.0),
                angular: ga::BiVec4::new(200.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            },
            inertia: Inertia::cuboid(ga::Vec4::new(LENGTHS[0], LENGTHS[1], LENGTHS[2], 1.0), 5.0),
            forque: Forque {
                linear: ga::Vec4::new(0.0, -200000.0, 0.0, 0.0),
                angular: ga::BiVec4::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        }
    ));

    // commands.spawn((
    //     Mesh3d(meshes.add(Torus::new(27.0, 33.0))),
    //     MeshMaterial3d(materials.add(Color::srgba_u8(164, 194, 255, 150))),
    //     Transform::from_xyz(0.0, 0.0, 0.0)
    // ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(30.0, 1.0, 30.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(164, 254, 150))),
        Transform::from_xyz(0.0, -10.5, 0.0)
    ));

    // light
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(20.0, 30.0, 40.0).looking_at(Vec3::ZERO, Vec3::Y),
        // The default cascade config is designed to handle large scenes.
        // As this example has a much smaller world, we can tighten the shadow
        // bounds for better visual quality.
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 4.0,
            maximum_distance: 10.0,
            ..default()
        }
            .build(),
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Projection::from(OrthographicProjection {
            // 6 world units per pixel of window height.
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 20.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(30.0, -5.0, 0.0).looking_at(-5.0 * Vec3::Y, Vec3::Y),
    ));
}

fn physics_update(mut query: Query<(&mut Transform, &mut Object)>, mut gizmos: Gizmos) {
    let (mut transform, mut object_mut): (Mut<Transform>, Mut<Object>) = query.single_mut();
    let object: &mut Object = &mut *object_mut;
    let proj_fn = proj_torus(30.0, 3.0);//proj_sphere(4.0);

    for _ in 0..N_SUBSTEPS {
        println!(
            "pose.ori: {:?}\n  grade0: {}\n  grade2: {}\n  grade4: {}",
            object.pose.ori,
            object.pose.ori.select_grade::<Const<0>>(),
            object.pose.ori.select_grade::<Const<2>>(),
            object.pose.ori.select_grade::<Const<4>>()
        );
        integrate_rate(&object.pose, &mut object.rate, &object.inertia, &object.forque, DT / N_SUBSTEPS as f64);
        // project_onto_surface(&mut object.pose, &mut object.rate, &proj_fn);
        cuboid_solve_wall_collision(
            ga::Vec4::new(0.0, -10.0, 0.0, 0.0),
            ga::Vec4::new(0.0, 1.0, 0.0, 0.0),
            &mut object.pose,
            &mut object.rate,
            &object.inertia,
            0.5,
            1.0,
            ga::Vec4::new(LENGTHS[0], LENGTHS[1], LENGTHS[2], 1.0),
            DT / N_SUBSTEPS as f64,
            &mut gizmos
        );
        integrate_pose(&mut object.pose, &mut object.rate, DT / N_SUBSTEPS as f64);
    }

    let pos = Vec3::from_slice(&[
        object.pose.pos[0] as f32,
        object.pose.pos[1] as f32,
        object.pose.pos[2] as f32,
    ]);

    let (_, normal) = proj_fn(&object.pose.pos);

    let normal = Vec3::from_slice(&[
        normal[0] as f32,
        normal[1] as f32,
        normal[2] as f32,
    ]);

    gizmos.arrow(Vec3::ZERO, pos, RED);
    gizmos.arrow(pos, pos + 2.0 * normal, RED);
    transform.translation = pos;

    let x_rot = object.pose.ori.rot(ga::Vec4::<FLOAT>::basis(0));
    let y_rot = object.pose.ori.rot(ga::Vec4::<FLOAT>::basis(1));
    let z_rot = object.pose.ori.rot(ga::Vec4::<FLOAT>::basis(2));

    let rot_matrix = Mat3::from_cols_array(&[
         x_rot[0] as f32, x_rot[1] as f32,  x_rot[2] as f32,
        y_rot[0] as f32,  y_rot[1] as f32,  y_rot[2] as f32,
         z_rot[0] as f32,  z_rot[1] as f32,  z_rot[2] as f32,
    ]);

    // println!("-- [X] --\n  LA: {}\n  GA: {}", rot_matrix * Vec3::X, object.pose.ori.rot(ga::Vec4::<FLOAT>::basis(0)));
    // println!("-- [Y] --\n  LA: {}\n  GA: {}", rot_matrix * Vec3::Y, object.pose.ori.rot(ga::Vec4::<FLOAT>::basis(1)));
    // println!("-- [Z] --\n  LA: {}\n  GA: {}", rot_matrix * Vec3::Z, object.pose.ori.rot(ga::Vec4::<FLOAT>::basis(2)));
    // println!("w~GA: {}", object.pose.ori.rot(ga::Vec4::<FLOAT>::basis(3)));
    gizmos.arrow(pos, pos + 2.0 * rot_matrix * Vec3::X, GREEN);

    transform.rotation = Quat::from_mat3(&rot_matrix)
}