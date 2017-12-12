/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


// Had to commented out the normalization, otherwise the simulator reports
// larger than expected yaw error.
static inline double normalize_angle(double theta) {
	// theta  = fmod(theta, 2.0 * M_PI);
    // if (theta > M_PI) {
    //   theta = 2.0 * M_PI - theta;
    // }
	return theta;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Create a normal (Gaussian) distribution for x, y, and theta respectively.
  random_device rd;
  mt19937 gen(rd());

  normal_distribution<double> dist_x(0.0, std[0]);
  normal_distribution<double> dist_y(0.0, std[1]);
  normal_distribution<double> dist_theta(0.0, std[2]);

  // set number of particles (to be tested)
  num_particles = 128;
  particles.resize(num_particles);
  weights.resize(num_particles);

  double sample_x, sample_y, sample_theta;

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id     = i;
    p.x      = x + dist_x(gen);
    p.y      = y + dist_y(gen);
    p.theta  = normalize_angle(theta + dist_theta(gen));
    p.weight = 1.0;

    particles[i] = p;
    weights[i]   = 1.0 / num_particles;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t,
                                double std_pos[],
                                double velocity,
                                double yaw_rate) {
  random_device rd;
  mt19937 gen(rd());
  double x, y, theta, delta_theta;

  // Gaussian noise
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    // update through motion model
    x     = particles[i].x;
    y     = particles[i].y;
    theta = particles[i].theta;

    if (fabs(yaw_rate) < 0.001) {
      x += velocity * delta_t * cos(theta);
      y += velocity * delta_t * sin(theta);
    } else {
      delta_theta = yaw_rate * delta_t;
      x          += velocity / yaw_rate * (sin(theta + delta_theta) - sin(theta));
      y          += velocity / yaw_rate * (cos(theta) - cos(theta + delta_theta));
      theta      += delta_theta;
    }

    particles[i].x     = x + dist_x(gen);
    particles[i].y     = y + dist_y(gen);
    particles[i].theta = normalize_angle(theta + dist_theta(gen));
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>  landmarks,
                                     std::vector<LandmarkObs>& observations) {
  double min_dist;
  double temp_dist;

  for (auto& obs: observations) {
    min_dist  = std::numeric_limits<double>::max();
    temp_dist = 0.0;

    for (const auto& landmark: landmarks) {
      temp_dist = dist(obs.x, obs.y, landmark.x, landmark.y);

      if (temp_dist < min_dist) {
        min_dist = temp_dist;
        obs.id   = landmark.id;
        // std::cout << "mis dist: " << min_dist << std::endl;
      }
    }
  }
}

void ParticleFilter::updateWeights(double                          sensor_range,
                                   double                          std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map                     & map_landmarks)
{
  // current particle states
  double x_p, y_p, theta_p;

  double gauss_norm, exponent;
  double mu_x, mu_y;

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double weight_p;

  std::vector<int>    associations_p;
  std::vector<double> sense_x_p;
  std::vector<double> sense_y_p;

  std::vector<LandmarkObs> landmarks_in_range_p;
  std::vector<LandmarkObs> observations_in_map_p;

  for (int i = 0; i < num_particles; i++) {
    x_p      = particles[i].x;
    y_p      = particles[i].y;
    theta_p  = particles[i].theta;
    weight_p = 1.0;

    landmarks_in_range_p.clear();
    observations_in_map_p.clear();

    associations_p.clear();
    sense_x_p.clear();
    sense_y_p.clear();

    // find all the landmarks within the sensor's range for the current particle
    for (const auto& l: map_landmarks.landmark_list) {
      if (dist(x_p, y_p, l.x_f, l.y_f) <= sensor_range) {
        LandmarkObs landmark;
        landmark.id = l.id_i;
        landmark.x  = l.x_f;
        landmark.y  = l.y_f;
        landmarks_in_range_p.push_back(landmark);
      }
    }

	if (landmarks_in_range_p.empty()) {
		particles[i].weight = 0.0;
		weights[i]          = 0.0;
		continue;
	}

    // convert observations to map coordinates for the current particle
    for (const auto& obs: observations) {
      // not sure this check is really needed.
      if (dist(0.0, 0.0, obs.x, obs.y) <= sensor_range) {
        LandmarkObs obs_in_map;
        obs_in_map.id = -1;
        obs_in_map.x  = x_p + cos(theta_p) * obs.x - sin(theta_p) * obs.y;
        obs_in_map.y  = y_p + sin(theta_p) * obs.x + cos(theta_p) * obs.y;
        observations_in_map_p.push_back(obs_in_map);
      }
    }

    // std::cout << "landmarks within range: " << landmarks_in_range_p.size() <<
    //   " observations: " << observations_in_map_p.size() << std::endl;

    // figure out corresponding landmark for each observation for the particle
    dataAssociation(landmarks_in_range_p, observations_in_map_p);

    for (const auto& obs: observations_in_map_p) {
      // get landmark position in map coordinate
      for (const auto& landmark: landmarks_in_range_p) {
        if (obs.id == landmark.id) {
          mu_x = landmark.x;
          mu_y = landmark.y;
          break;
        }
      }

      // Calculate how likely a measurement should be and combine them together
      gauss_norm = (1.0 / (2 * M_PI * sig_x * sig_y));
      exponent   = ((obs.x - mu_x) * (obs.x - mu_x)) / (2 * sig_x * sig_x) +
                   ((obs.y - mu_y) * (obs.y - mu_y)) / (2 * sig_y * sig_y);

      weight_p *= (gauss_norm * exp(-exponent));

      // record association for the particle
      associations_p.push_back(obs.id);
      sense_x_p.push_back(obs.x);
      sense_y_p.push_back(obs.y);
    }

    // set association for the particle
    SetAssociations(particles[i], associations_p, sense_x_p, sense_y_p);

    // record un-normalized importance weight for each particle
    particles[i].weight = weight_p;
    weights[i]          = weight_p;
  }
}

void ParticleFilter::resample() {
  // renormalize the weights vector. There is no need to do it
  // since std::discrete_distribution alreday handles it internally.
  // double sum_of_weights = accumulate(weights.begin(), weights.end(), 0.0);
  // transform(weights.begin(), weights.end(), weights.begin(),
  // bind1st(divides<double>(), sum_of_weights));

  // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  random_device rd;
  mt19937 gen(rd());

  discrete_distribution<> d(weights.begin(), weights.end());
  vector<Particle> new_particles(num_particles);
  int index;

  for (int i = 0; i < num_particles; i++) {
    index            = d(gen);
    new_particles[i] = particles[index];
  }

  particles.swap(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle           particle,
                                         std::vector<int>   associations,
                                         std::vector<double>sense_x,
                                         std::vector<double>sense_y)
{
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x      = sense_x;
  particle.sense_y      = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int>  v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream   ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream   ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
