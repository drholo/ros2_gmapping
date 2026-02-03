#include "gmapper/mapper.h"

#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

using std::placeholders::_1;

SlamGmapping::SlamGmapping():
    Node("slam_gmapping"),
    scan_filter_sub_(nullptr),
    scan_filter_(nullptr),
    laser_count_(0),
    transform_thread_(nullptr)
{
    buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
     auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        get_node_base_interface(),
        get_node_timers_interface());
    buffer_->setCreateTimerInterface(timer_interface);
    tfl_ = std::make_shared<tf2_ros::TransformListener>(*buffer_);
    node_ = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node *) {});
    tfB_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    map_to_odom_.setIdentity();
    seed_ = static_cast<unsigned long>(time(nullptr));
    last_scan_time_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    have_last_scan_time_ = false;
    last_tf_publish_time_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    have_last_tf_publish_time_ = false;
    
    // Declare and get config_path parameter
    std::string config_file = this->declare_parameter("config_file", "config/gmapping_params.yaml");
    
    init(config_file);
    startLiveSlam();
}

void SlamGmapping::init(const std::string& config_file) {
    gsp_ = new GMapping::GridSlamProcessor();

    gsp_laser_ = nullptr;
    gsp_odom_ = nullptr;
    got_first_scan_ = false;
    got_map_ = false;

    // Determine the full path to the config file
    std::string full_config_path;
    if (config_file[0] == '/' || config_file[0] == '~') {
        // Absolute path provided
        full_config_path = config_file;
    } else {
        // Relative path - resolve from package share directory
        try {
            std::string package_share_dir = ament_index_cpp::get_package_share_directory("gmapper");
            full_config_path = package_share_dir + "/" + config_file;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to find package share directory: %s", e.what());
            full_config_path = config_file;
        }
    }

    // Load parameters from YAML config file
    YAML::Node config;
    try {
        config = YAML::LoadFile(full_config_path);
        RCLCPP_INFO(this->get_logger(), "Loaded parameters from config file: %s", full_config_path.c_str());
    } catch (const YAML::Exception& e) {
        RCLCPP_WARN(this->get_logger(), 
            "Failed to load config file %s: %s. Using default parameters.",
            full_config_path.c_str(), e.what());
        config = YAML::Node();
    }

    // Helper lambda to get parameter value from YAML with fallback to default
    auto get_param = [&](const std::string& param_name, const YAML::Node& default_value) {
        try {
            if (config["slam_gmapping"] && config["slam_gmapping"]["ros__parameters"]) {
                YAML::Node param = config["slam_gmapping"]["ros__parameters"][param_name];
                if (param.IsDefined()) {
                    return param;
                }
            }
        } catch (const YAML::Exception& e) {
            RCLCPP_DEBUG(this->get_logger(), "Error reading parameter %s from config: %s",
                param_name.c_str(), e.what());
        }
        return default_value;
    };

    // Frame names and basic settings
    throttle_scans_ = get_param("throttle_scans", YAML::Node(1)).as<int>();
    base_frame_ = get_param("base_frame", YAML::Node("base_link")).as<std::string>();
    map_frame_ = get_param("map_frame", YAML::Node("map")).as<std::string>();
    odom_frame_ = get_param("odom_frame", YAML::Node("odom")).as<std::string>();
    transform_publish_period_ = get_param("transform_publish_period", YAML::Node(0.05)).as<double>();

    double map_update_interval_sec = get_param("map_update_interval", YAML::Node(0.5)).as<double>();
    map_update_interval_ = tf2::durationFromSec(map_update_interval_sec);

    // Range parameters
    maxUrange_ = get_param("maxUrange", YAML::Node(80.0)).as<double>();
    maxRange_ = get_param("maxRange", YAML::Node(0.0)).as<double>();

    // Scan matching parameters
    minimum_score_ = get_param("minimum_score", YAML::Node(0.0)).as<double>();
    sigma_ = get_param("sigma", YAML::Node(0.05)).as<double>();
    kernelSize_ = get_param("kernelSize", YAML::Node(1)).as<int>();
    lstep_ = get_param("lstep", YAML::Node(0.05)).as<double>();
    astep_ = get_param("astep", YAML::Node(0.05)).as<double>();
    iterations_ = get_param("iterations", YAML::Node(5)).as<int>();
    lsigma_ = get_param("lsigma", YAML::Node(0.075)).as<double>();
    ogain_ = get_param("ogain", YAML::Node(3.0)).as<double>();
    lskip_ = get_param("lskip", YAML::Node(0)).as<int>();

    // Motion model parameters
    srr_ = get_param("srr", YAML::Node(0.1)).as<double>();
    srt_ = get_param("srt", YAML::Node(0.2)).as<double>();
    str_ = get_param("str", YAML::Node(0.1)).as<double>();
    stt_ = get_param("stt", YAML::Node(0.2)).as<double>();

    // Update parameters
    linearUpdate_ = get_param("linearUpdate", YAML::Node(1.0)).as<double>();
    angularUpdate_ = get_param("angularUpdate", YAML::Node(0.5)).as<double>();
    temporalUpdate_ = get_param("temporalUpdate", YAML::Node(1.0)).as<double>();
    resampleThreshold_ = get_param("resampleThreshold", YAML::Node(0.5)).as<double>();

    // Particle filter
    particles_ = get_param("particles", YAML::Node(30)).as<int>();

    // Map parameters
    xmin_ = get_param("xmin", YAML::Node(-10.0)).as<double>();
    ymin_ = get_param("ymin", YAML::Node(-10.0)).as<double>();
    xmax_ = get_param("xmax", YAML::Node(10.0)).as<double>();
    ymax_ = get_param("ymax", YAML::Node(10.0)).as<double>();
    delta_ = get_param("delta", YAML::Node(0.05)).as<double>();
    occ_thresh_ = get_param("occ_thresh", YAML::Node(0.25)).as<double>();

    // Likelihood sampling
    llsamplerange_ = get_param("llsamplerange", YAML::Node(0.01)).as<double>();
    llsamplestep_ = get_param("llsamplestep", YAML::Node(0.01)).as<double>();
    lasamplerange_ = get_param("lasamplerange", YAML::Node(0.005)).as<double>();
    lasamplestep_ = get_param("lasamplestep", YAML::Node(0.005)).as<double>();

    tf_delay_ = transform_publish_period_;
}

void SlamGmapping::startLiveSlam() {
    entropy_publisher_ = this->create_publisher<std_msgs::msg::Float64>("entropy", rclcpp::SystemDefaultsQoS());
    sst_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map", rclcpp::SystemDefaultsQoS());
    sstm_ = this->create_publisher<nav_msgs::msg::MapMetaData>("map_metadata", rclcpp::SystemDefaultsQoS());
    scan_filter_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>
            (node_, "scan", rclcpp::SensorDataQoS().get_rmw_qos_profile());
//    sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
//        "scan", rclcpp::SensorDataQoS(),
//        std::bind(&SlamGmapping::laserCallback, this, std::placeholders::_1));
    scan_filter_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>>
            (*scan_filter_sub_, *buffer_, odom_frame_, 10, node_);
    scan_filter_->registerCallback(std::bind(&SlamGmapping::laserCallback, this, std::placeholders::_1));
    transform_thread_ = std::make_shared<std::thread>
            (std::bind(&SlamGmapping::publishLoop, this, transform_publish_period_));
}

void SlamGmapping::publishLoop(double transform_publish_period){
    if (transform_publish_period == 0)
        return;
    rclcpp::Rate r(1.0 / transform_publish_period);
    while (rclcpp::ok()) {
        publishTransform();
        r.sleep();
    }
}

SlamGmapping::~SlamGmapping()
{
    if(transform_thread_){
        transform_thread_->join();
    }

    delete gsp_;
    delete gsp_laser_;
    delete gsp_odom_;
}

bool SlamGmapping::getOdomPose(GMapping::OrientedPoint& gmap_pose, const rclcpp::Time& t)
{
    // Get the pose of the centered laser at the right time
    centered_laser_pose_.header.stamp = t;
    // Get the laser's pose that is centered
    geometry_msgs::msg::PoseStamped odom_pose;
    try
    {
        buffer_->transform(centered_laser_pose_, odom_pose, odom_frame_, tf2::durationFromSec(1.0));
    }
    catch(tf2::TransformException& e)
    {
        RCLCPP_WARN(this->get_logger(), "Failed to compute odom pose, skipping scan (%s)", e.what());
        return false;
    }

    double yaw = tf2::getYaw(odom_pose.pose.orientation);

    gmap_pose = GMapping::OrientedPoint(odom_pose.pose.position.x,
                                        odom_pose.pose.position.y,
                                        yaw);
    return true;
}

bool SlamGmapping::initMapper(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
{
    laser_frame_ = scan->header.frame_id;
    // Get the laser's pose, relative to base.
    geometry_msgs::msg::PoseStamped ident;
    geometry_msgs::msg::PoseStamped laser_pose;

    try{
        ident.header.frame_id = laser_frame_;
        ident.header.stamp = scan->header.stamp;
        tf2::Transform transform;
        transform.setIdentity();
        tf2::toMsg(transform, ident.pose);
        buffer_->transform(ident, laser_pose, base_frame_);
    }
    catch (tf2::TransformException& e){
        RCLCPP_WARN(this->get_logger(), "Failed to compute laser pose, aborting initialization (%s)", e.what());
        return false;
    }

    // create a point 1m above the laser position and transform it into the laser-frame
    geometry_msgs::msg::PointStamped up;
    up.header.stamp = scan->header.stamp;
    up.header.frame_id = base_frame_;
    up.point.x = up.point.y = 0;
    up.point.z = 1 + laser_pose.pose.position.z;
    try
    {
        buffer_->transform(up, up, laser_frame_);
    }
    catch(tf2::TransformException& e)
    {
        RCLCPP_WARN(this->get_logger(), "Unable to determine orientation of laser: %s", e.what());
        return false;
    }

    // gmapping doesnt take roll or pitch into account. So check for correct sensor alignment.
    if (fabs(fabs(up.point.z) - 1) > 0.001)
    {
        RCLCPP_INFO(this->get_logger(),
                "Laser has to be mounted planar! Z-coordinate has to be 1 or -1, but gave: %.5f", up.point.z);
        return false;
    }

    gsp_laser_beam_count_ = static_cast<unsigned int>(scan->ranges.size());

    double angle_center = (scan->angle_min + scan->angle_max)/2;

    centered_laser_pose_.header.frame_id = laser_frame_;
    centered_laser_pose_.header.stamp = get_clock()->now();
    tf2::Quaternion q;

    if (up.point.z > 0)
    {
        do_reverse_range_ = scan->angle_min > scan->angle_max;
        q.setEuler(angle_center, 0, 0);
        RCLCPP_INFO(this->get_logger(),"Laser is mounted upwards.");
    }
    else
    {
        do_reverse_range_ = scan->angle_min < scan->angle_max;
        q.setEuler(-angle_center, 0, M_PI);
        RCLCPP_INFO(this->get_logger(), "Laser is mounted upside down.");
    }

    centered_laser_pose_.pose.position.x = 0;
    centered_laser_pose_.pose.position.y = 0;
    centered_laser_pose_.pose.position.z = 0;

    centered_laser_pose_.pose.orientation.w = q.getW();
    centered_laser_pose_.pose.orientation.x = q.getX();
    centered_laser_pose_.pose.orientation.y = q.getY();
    centered_laser_pose_.pose.orientation.z = q.getZ();

    // Compute the angles of the laser from -x to x, basically symmetric and in increasing order
    laser_angles_.resize(scan->ranges.size());
    // Make sure angles are started so that they are centered
    double theta = - std::fabs(scan->angle_min - scan->angle_max)/2;
    for(unsigned int i=0; i<scan->ranges.size(); ++i)
    {
        laser_angles_[i]=theta;
        theta += std::fabs(scan->angle_increment);
    }

    RCLCPP_DEBUG(this->get_logger(), "Laser angles in laser-frame: min: %.3f max: %.3f inc: %.3f",
            scan->angle_min, scan->angle_max, scan->angle_increment);
    RCLCPP_DEBUG(this->get_logger(), "Laser angles in top-down centered laser-frame: min: %.3f max: %.3f inc: %.3f",
            laser_angles_.front(), laser_angles_.back(), std::fabs(scan->angle_increment));

    GMapping::OrientedPoint gmap_pose(0, 0, 0);

    // setting maxRange and maxUrange here so we can set a reasonable default
    maxRange_ = scan->range_max - 0.01;
    maxUrange_ = maxRange_;

    // The laser must be called "FLASER".
    // We pass in the absolute value of the computed angle increment, on the
    // assumption that GMapping requires a positive angle increment.  If the
    // actual increment is negative, we'll swap the order of ranges before
    // feeding each scan to GMapping.
    gsp_laser_ = new GMapping::RangeSensor("FLASER", gsp_laser_beam_count_, fabs(scan->angle_increment), gmap_pose,
                                           0.0, maxRange_);

    GMapping::SensorMap smap;
    smap.insert(make_pair(gsp_laser_->getName(), gsp_laser_));
    gsp_->setSensorMap(smap);

    gsp_odom_ = new GMapping::OdometrySensor(odom_frame_);

    /// @todo Expose setting an initial pose
    GMapping::OrientedPoint initialPose;
    if(!getOdomPose(initialPose, scan->header.stamp))
    {
        RCLCPP_WARN(this->get_logger(), "Unable to determine inital pose of laser! Starting point will be set to zero.");
        initialPose = GMapping::OrientedPoint(0.0, 0.0, 0.0);
    }

    gsp_->setMatchingParameters(maxUrange_, maxRange_, sigma_,
                                kernelSize_, lstep_, astep_, iterations_,
                                lsigma_, ogain_, static_cast<unsigned int>(lskip_));

    gsp_->setMotionModelParameters(srr_, srt_, str_, stt_);
    gsp_->setUpdateDistances(linearUpdate_, angularUpdate_, resampleThreshold_);
    gsp_->setUpdatePeriod(temporalUpdate_);
    gsp_->setgenerateMap(false);
    gsp_->GridSlamProcessor::init(static_cast<unsigned int>(particles_), xmin_, ymin_, xmax_, ymax_,
                                  delta_, initialPose);
    gsp_->setllsamplerange(llsamplerange_);
    gsp_->setllsamplestep(llsamplestep_);
    /// @todo Check these calls; in the gmapping gui, they use
    /// llsamplestep and llsamplerange intead of lasamplestep and
    /// lasamplerange.  It was probably a typo, but who knows.
    gsp_->setlasamplerange(lasamplerange_);
    gsp_->setlasamplestep(lasamplestep_);
    gsp_->setminimumScore(minimum_score_);

    // Call the sampling function once to set the seed.
    GMapping::sampleGaussian(1, static_cast<unsigned int>(seed_));

    RCLCPP_INFO(this->get_logger(), "Initialization complete");

    return true;
}

bool SlamGmapping::addScan(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan, GMapping::OrientedPoint& gmap_pose) {
    if (!getOdomPose(gmap_pose, scan->header.stamp))
        return false;

    if (scan->ranges.size() != gsp_laser_beam_count_)
        return false;

    // GMapping wants an array of doubles...
    auto *ranges_double = new double[scan->ranges.size()];
    // If the angle increment is negative, we have to invert the order of the readings.
    if (do_reverse_range_) {
        RCLCPP_DEBUG(this->get_logger(), "Inverting scan");
        int num_ranges = static_cast<int>(scan->ranges.size());
        for (int i = 0; i < num_ranges; i++) {
            // Must filter out short readings, because the mapper won't
            if (scan->ranges[num_ranges - i - 1] < scan->range_min)
                ranges_double[i] = (double) scan->range_max;
            else
                ranges_double[i] = (double) scan->ranges[num_ranges - i - 1];
        }
    } else {
        for (unsigned int i = 0; i < scan->ranges.size(); i++) {
            // Must filter out short readings, because the mapper won't
            if (scan->ranges[i] < scan->range_min)
                ranges_double[i] = (double) scan->range_max;
            else
                ranges_double[i] = (double) scan->ranges[i];
        }
    }

    GMapping::RangeReading reading(static_cast<unsigned int>(scan->ranges.size()),
                                   ranges_double,
                                   gsp_laser_,
                                   scan->header.stamp.sec);

    // ...but it deep copies them in RangeReading constructor, so we don't
    // need to keep our array around.
    delete[] ranges_double;

    reading.setPose(gmap_pose);

    RCLCPP_DEBUG(this->get_logger(), "processing scan");

    return gsp_->processScan(reading);
}


void SlamGmapping::laserCallback(sensor_msgs::msg::LaserScan::ConstSharedPtr scan) {
    laser_count_++;
    if ((laser_count_ % throttle_scans_) != 0)
        return;

    tf2::TimePoint last_map_update = tf2::TimePointZero;

    // We can't initialize the mapper until we've got the first scan
    if(!got_first_scan_)
    {
        if(!initMapper(scan))
            return;
        got_first_scan_ = true;
    }

    GMapping::OrientedPoint odom_pose;

    if(addScan(scan, odom_pose))
    {
        GMapping::OrientedPoint mpose = gsp_->getParticles()[gsp_->getBestParticleIndex()].pose;

        tf2::Quaternion q;
        q.setRPY(0, 0, mpose.theta);
        tf2::Transform laser_to_map = tf2::Transform(q, tf2::Vector3(mpose.x, mpose.y, 0.0)).inverse();
        q.setRPY(0, 0, odom_pose.theta);
        tf2::Transform odom_to_laser = tf2::Transform(q, tf2::Vector3(odom_pose.x, odom_pose.y, 0.0));

        map_to_odom_mutex_.lock();
        map_to_odom_ = (odom_to_laser * laser_to_map).inverse();
        last_scan_time_ = scan->header.stamp;
        have_last_scan_time_ = true;
        map_to_odom_mutex_.unlock();

        tf2::TimePoint timestamp = tf2_ros::fromMsg(scan->header.stamp);
        if(!got_map_ || (timestamp - last_map_update) > map_update_interval_)
        {
            updateMap(scan);
            last_map_update = tf2_ros::fromMsg(scan->header.stamp);
        }
    }
}

double SlamGmapping::computePoseEntropy()
{
    double weight_total=0.0;
    for (const auto &it : gsp_->getParticles()) {
        weight_total += it.weight;
    }
    double entropy = 0.0;
    for (const auto &it : gsp_->getParticles()) {
        if(it.weight/weight_total > 0.0)
            entropy += it.weight/weight_total * log(it.weight/weight_total);
    }
    return -entropy;
}

void SlamGmapping::updateMap(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
{
    RCLCPP_DEBUG(this->get_logger(), "Update map");
    map_mutex_.lock();
    GMapping::ScanMatcher matcher;

    matcher.setLaserParameters(static_cast<unsigned int>(scan->ranges.size()), &(laser_angles_[0]),
                               gsp_laser_->getPose());

    matcher.setlaserMaxRange(maxRange_);
    matcher.setusableRange(maxUrange_);
    matcher.setgenerateMap(true);

    GMapping::GridSlamProcessor::Particle best =
            gsp_->getParticles()[gsp_->getBestParticleIndex()];
    std_msgs::msg::Float64 entropy;
    entropy.data = computePoseEntropy();
    if(entropy.data > 0.0)
        entropy_publisher_->publish(entropy);

    if(!got_map_) {
        map_.info.resolution = static_cast<nav_msgs::msg::MapMetaData::_resolution_type>(delta_);
        map_.info.origin.position.x = 0.0;
        map_.info.origin.position.y = 0.0;
        map_.info.origin.position.z = 0.0;
        map_.info.origin.orientation.x = 0.0;
        map_.info.origin.orientation.y = 0.0;
        map_.info.origin.orientation.z = 0.0;
        map_.info.origin.orientation.w = 1.0;
    }

    GMapping::Point center;
    center.x=(xmin_ + xmax_) / 2.0;
    center.y=(ymin_ + ymax_) / 2.0;

    GMapping::ScanMatcherMap smap(center, xmin_, ymin_, xmax_, ymax_,
                                  delta_);

    RCLCPP_DEBUG(this->get_logger(), "Trajectory tree:");
    for(GMapping::GridSlamProcessor::TNode* n = best.node;
        n;
        n = n->parent)
    {
        RCLCPP_DEBUG(this->get_logger(), "  %.3f %.3f %.3f",
                  n->pose.x,
                  n->pose.y,
                  n->pose.theta);
        if(!n->reading)
        {
            RCLCPP_DEBUG(this->get_logger(), "Reading is NULL");
            continue;
        }
        matcher.invalidateActiveArea();
        matcher.computeActiveArea(smap, n->pose, &((*n->reading)[0]));
        matcher.registerScan(smap, n->pose, &((*n->reading)[0]));
    }

    // the map may have expanded, so resize ros message as well
    if(map_.info.width != (unsigned int) smap.getMapSizeX() || map_.info.height != (unsigned int) smap.getMapSizeY()) {

        // NOTE: The results of ScanMatcherMap::getSize() are different from the parameters given to the constructor
        //       so we must obtain the bounding box in a different way
        GMapping::Point wmin = smap.map2world(GMapping::IntPoint(0, 0));
        GMapping::Point wmax = smap.map2world(GMapping::IntPoint(smap.getMapSizeX(), smap.getMapSizeY()));
        xmin_ = wmin.x; ymin_ = wmin.y;
        xmax_ = wmax.x; ymax_ = wmax.y;

        RCLCPP_DEBUG(this->get_logger(), "map size is now %dx%d pixels (%f,%f)-(%f, %f)", smap.getMapSizeX(), smap.getMapSizeY(),
                  xmin_, ymin_, xmax_, ymax_);

        map_.info.width = static_cast<nav_msgs::msg::MapMetaData::_width_type>(smap.getMapSizeX());
        map_.info.height = static_cast<nav_msgs::msg::MapMetaData::_height_type>(smap.getMapSizeY());
        map_.info.origin.position.x = xmin_;
        map_.info.origin.position.y = ymin_;
        map_.data.resize(map_.info.width * map_.info.height);

        RCLCPP_DEBUG(this->get_logger(), "map origin: (%f, %f)", map_.info.origin.position.x, map_.info.origin.position.y);
    }

    for(int x=0; x < smap.getMapSizeX(); x++)
    {
        for(int y=0; y < smap.getMapSizeY(); y++)
        {
            /// @todo Sort out the unknown vs. free vs. obstacle thresholding
            GMapping::IntPoint p(x, y);
            double occ=smap.cell(p);
            assert(occ <= 1.0);
            if(occ < 0)
                map_.data[MAP_IDX(map_.info.width, x, y)] = -1;
            else if(occ > occ_thresh_)
            {
                //map_.map.data[MAP_IDX(map_.map.info.width, x, y)] = (int)round(occ*100.0);
                map_.data[MAP_IDX(map_.info.width, x, y)] = 100;
            }
            else
                map_.data[MAP_IDX(map_.info.width, x, y)] = 0;
        }
    }
    got_map_ = true;

    //make sure to set the header information on the map
    map_.header.stamp = get_clock()->now();
    map_.header.frame_id = map_frame_;

    sst_->publish(map_);
    sstm_->publish(map_.info);
    map_mutex_.unlock();
}

void SlamGmapping::publishTransform()
{
    if (!have_last_scan_time_) {
        return;
    }
    map_to_odom_mutex_.lock();
    rclcpp::Time base_time = have_last_scan_time_ ? last_scan_time_ : get_clock()->now();
    if (have_last_tf_publish_time_ && base_time <= last_tf_publish_time_) {
        base_time = last_tf_publish_time_ + rclcpp::Duration::from_seconds(1e-6);
    }
    rclcpp::Time tf_expiration = base_time + rclcpp::Duration::from_seconds(tf_delay_);
    last_tf_publish_time_ = base_time;
    have_last_tf_publish_time_ = true;
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = map_frame_;
    transform.header.stamp = tf_expiration;
    transform.child_frame_id = odom_frame_;
    try {
        transform.transform = tf2::toMsg(map_to_odom_);
        tfB_->sendTransform(transform);
    }
    catch (tf2::LookupException& te){
        RCLCPP_INFO(this->get_logger(), te.what());
    }
    map_to_odom_mutex_.unlock();
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto slam_gmapping_node = std::make_shared<SlamGmapping>();
    rclcpp::spin(slam_gmapping_node);
    return(0);
}