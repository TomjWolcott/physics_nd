//! A simple 3D scene with light shining over a cube sitting on a plane.

use std::any::{type_name, type_name_of_val};
use bevy::prelude::*;
use physics_nd::components::*;
use physics_nd::*;
use wedged::algebra as ga;

const DT: f64 = 0.001;
const N_SUBSTEPS: usize = 1;

fn main() {
    println!("Hello, world!");
    App::new()
        .add_plugins(DefaultPlugins)
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

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.3, 2.0, 4.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(164, 194, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Object {
            pose: Pose::default(),
            rate: Rate {
                linear: ga::Vec4::new(0.0, 0.0, 0.0, 0.0),
                angular: ga::BiVec4::new(60.0, 0.0, 30.1, 0.0, 0.0, 0.0)
            },
            inertia: Inertia::cuboid(ga::Vec4::new(0.3, 2.0, 4.0, 1.0), 5.0),
            forque: Forque {
                linear: ga::Vec4::new(0.0, 0.0, 0.0, 0.0),
                angular: ga::BiVec4::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        }
    ));

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn physics_update(mut query: Query<(&mut Transform, &mut Object)>) {
    let (mut transform, mut object_mut): (Mut<Transform>, Mut<Object>) = query.single_mut();
    let object: &mut Object = &mut *object_mut;

    println!("rate: {:?}\ninertia tensor: {}", object.rate, object.inertia.inertia_tensor);

    for _ in 0..N_SUBSTEPS {
        integrate(&mut object.pose, &mut object.rate, &object.inertia, &object.forque, DT / N_SUBSTEPS as f64);
    }

    let pos = [
        object.pose.pos[0] as f32,
        object.pose.pos[1] as f32,
        object.pose.pos[2] as f32,
    ];
    transform.translation = Vec3::from_slice(&pos);

    let x_rot = object.pose.ori.rot(ga::Vec4::new(1.0, 0.0, 0.0, 0.0));
    let y_rot = object.pose.ori.rot(ga::Vec4::new(0.0, 1.0, 0.0, 0.0));
    let z_rot = object.pose.ori.rot(ga::Vec4::new(0.0, 0.0, 1.0, 0.0));

    let rot_matrix = [
        x_rot[0] as f32, x_rot[1] as f32, x_rot[2] as f32,
        y_rot[0] as f32, y_rot[1] as f32, y_rot[2] as f32,
        z_rot[0] as f32, z_rot[1] as f32, z_rot[2] as f32,
    ];

    transform.rotation = Quat::from_mat3(&Mat3::from_cols_array(&rot_matrix).transpose())
}