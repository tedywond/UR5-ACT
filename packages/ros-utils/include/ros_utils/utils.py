import rospy


def ROS_DEBUG( msg, *args ):
    return rospy.logdebug( msg, *args )


def ROS_INFO( msg, *args ):
    return rospy.loginfo( msg, *args )


def ROS_WARN( msg, *args ):
    return rospy.logwarn( msg, *args )


def ROS_ERR( msg, *args ):
    return rospy.logerr( msg, *args )


def ROS_FATAL( msg, *args ):
    return rospy.logfatal( msg, *args )
