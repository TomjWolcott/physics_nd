//! A simple 3D scene with light shining over a cube sitting on a plane.

use std::any::{type_name, type_name_of_val};
use bevy::color::palettes::basic::GREEN;
use bevy::color::palettes::css::{RED, WHITE};
use bevy::prelude::*;
use bevy::render::camera::ScalingMode;
use physics_nd::components::*;
use physics_nd::*;
use wedged::algebra as ga;

const DT: f64 = 0.001;
const N_SUBSTEPS: usize = 1;

fn main() {
    println!("Hello, world!");

    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(WHITE.into()))
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
        MeshMaterial3d(materials.add(Color::srgb_u8(244, 234, 155))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Object {
            pose: Pose {
                pos: ga::Vec4::new(2.0, 0.0, 0.0, 0.0),
                ..default()
            },
            rate: Rate {
                linear: ga::Vec4::new(0.0, -50.0, 30.0, 0.0),
                angular: ga::BiVec4::new(70.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            },
            inertia: Inertia::cuboid(ga::Vec4::new(0.3, 2.0, 4.0, 1.0), 5.0),
            forque: Forque {
                linear: ga::Vec4::new(0.0, -30000.0, 0.0, 0.0),
                angular: ga::BiVec4::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        }
    ));

    commands.spawn((
        Mesh3d(meshes.add(Torus::new(7.0, 13.0))),
        MeshMaterial3d(materials.add(Color::srgba_u8(164, 194, 255, 150))),
        Transform::from_xyz(0.0, 0.0, 0.0)
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
        Projection::from(OrthographicProjection {
            // 6 world units per pixel of window height.
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 20.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(10.0, 20.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn physics_update(mut query: Query<(&mut Transform, &mut Object)>, mut gizmos: Gizmos) {
    let (mut transform, mut object_mut): (Mut<Transform>, Mut<Object>) = query.single_mut();
    let object: &mut Object = &mut *object_mut;
    let proj_fn = proj_torus(10.0, 3.0);//proj_sphere(4.0);

    for _ in 0..N_SUBSTEPS {
        integrate(&mut object.pose, &mut object.rate, &object.inertia, &object.forque, DT / N_SUBSTEPS as f64);
        project_onto_surface(&mut object.pose, &mut object.rate, &proj_fn);
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