//! Convenient imports for the curated application API.

pub use crate::api::InputError;
pub use crate::api::pose::{DEFAULT_ROTATION_TOLERANCE, Pose, PoseError};
pub use crate::api::pose::{
    left_update as se3_left_update, right_point_action_jacobian, right_update as se3_right_update,
};
pub use crate::api::rotation::{left_update as so3_left_update, right_update as so3_right_update};
pub use crate::api::types::{Mat3, Mat3x6, Mat3x9, Mat6, Mat9, Point3, Twist, Vec3, Vec6, Vec9};

pub use crate::api::extended_pose::ExtendedPose;
pub use crate::api::extended_pose::{
    left_update as se23_left_update,
    right_point_action_position_jacobian as se23_right_point_action_position_jacobian,
    right_point_action_velocity_jacobian as se23_right_point_action_velocity_jacobian,
    right_update as se23_right_update,
};

pub use crate::api::quaternion_pose::{DEFAULT_QUATERNION_TOLERANCE, QuatPose, QuatPoseError};
pub use crate::api::quaternion_pose::{
    left_update as quat_se3_left_update,
    right_point_action_jacobian as quat_se3_right_point_action_jacobian,
    right_update as quat_se3_right_update,
};
