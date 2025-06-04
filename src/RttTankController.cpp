#include <cnoid/SimpleController>
#include <geometry_msgs/Twist.h>
#include <ros/node_handle.h>
#include <mutex>

using namespace cnoid;

class RttTankController : public SimpleController {
    std::unique_ptr<ros::NodeHandle> node;
    ros::Subscriber subscriber;
    geometry_msgs::Twist latest_command_velocity;
    std::mutex command_velocity_mutex;

    Link* trackL, *trackR;
    Link* turretJoint[2];

    double q_ref[2], q_prev[2];
    double dt;

public:
    virtual bool configure(SimpleControllerConfig* config) override {
        node.reset(new ros::NodeHandle);
        return true;
    }

    virtual bool initialize(SimpleControllerIO* io) override {
        std::ostream& os = io->os();
        Body* body = io->body();
        dt = io->timeStep();

        trackL = body->link("TRACK_L");
        trackR = body->link("TRACK_R");

        io->enableOutput(trackL, JointVelocity);
        io->enableOutput(trackR, JointVelocity);

        turretJoint[0] = body->link("TURRET_Y");
        turretJoint[1] = body->link("TURRET_P");

        for(int i=0; i < 2; ++i){
            Link* joint = turretJoint[i];
            q_ref[i] = q_prev[i] = joint->q();
            joint->setActuationMode(JointTorque);
            io->enableIO(joint);
        }

        subscriber = node->subscribe("cmd_vel", 1, &RttTankController::command_velocity_callback, this);    return true;
    }

    void command_velocity_callback(const geometry_msgs::Twist& msg) {
        std::lock_guard<std::mutex> lock(command_velocity_mutex);
        latest_command_velocity = msg;
    }

    virtual bool control() override {
        geometry_msgs::Twist command_velocity;
        {
            std::lock_guard<std::mutex> lock(command_velocity_mutex);
            command_velocity = latest_command_velocity;
        }
        // set the velocity of each tracks
        trackL->dq_target() = 0.5 * command_velocity.linear.x - 0.3*command_velocity.angular.z;
        trackR->dq_target() = 0.5 * command_velocity.linear.x + 0.3*command_velocity.angular.z;

        static const double P = 200.0;
        static const double D = 50.0;
        for(int i=0; i < 2; ++i){
            Link* joint = turretJoint[i];
            double q = joint->q();
            double dq = (q - q_prev[i]) / dt;
            double dq_ref = 0.0;
            joint->u() = P * (q_ref[i] - q) + D * (dq_ref - dq);
            q_prev[i] = q;
        }

        return true;
    }

    virtual void stop() override {
        subscriber.shutdown();
    }
};
CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(RttTankController)


