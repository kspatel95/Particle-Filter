/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 20;  // TODO: Set the number of particles
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  std::default_random_engine gen;
  for (int i=0; i < num_particles; i++) {
    // Initialized particles
    double sample_x = dist_x(gen);
    double sample_y = dist_y(gen);
    double sample_theta = dist_theta(gen);
    // Initialize weight
    double init_weight = 1.0;
    
    Particle particle = {
      i, // id
      sample_x, // x
      sample_y, // y
      sample_theta, // theta
      init_weight, // weight
      {}, // associations
      {}, // sense_x
      {}, // sense_y
    };
  
    // Add new particle to list of Particles
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  
  // Function does not return anything
  is_initialized = true;
  
  //std::cout << "-----INITIALIZATION DONE!-----" << std::endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // Random
  default_random_engine gen;
  
  for (int i=0; i < num_particles; i++) {
    // Gaussian noise setup
    int mean = 0;
    normal_distribution<double> noise_x(mean, std_pos[0]);
    normal_distribution<double> noise_y(mean, std_pos[1]);
    normal_distribution<double> noise_theta(mean, std_pos[2]);
    
    // Define predictions
    double x_pred, y_pred, theta_pred;
    
    // Define particle components
    double x_0 = particles[i].x;
    double y_0 = particles[i].y;
    double theta = particles[i].theta;
    
    // Avoid calculations by 0
    if (fabs(yaw_rate) > 0.01) {
      // Calculations
      x_pred = x_0 + (velocity / yaw_rate) * (sin(theta + (yaw_rate * delta_t)) - sin(theta));
      y_pred = y_0 + (velocity / yaw_rate) * (cos(theta) - cos(theta + (yaw_rate * delta_t)));
      theta_pred = theta + (yaw_rate * delta_t);
    }
    else {
      x_pred = x_0 + velocity * delta_t * cos(theta);
      y_pred = y_0 + velocity * delta_t * sin(theta);
      theta_pred = theta;
    }
    
    // Predictions with sensor noise
    particles[i].x = x_pred + noise_x(gen);
    particles[i].y = y_pred + noise_y(gen);
    particles[i].theta = theta_pred + noise_theta(gen);
  }
  //std::cout << "-----PREDICTION DONE!-----" << std::endl; // DEBUG
}

void ParticleFilter::dataAssociation(vector<LandmarkObs>& tf_observations, 
                                     vector<LandmarkObs> sense_range) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  // For each observation
  for (LandmarkObs& tf_obs : tf_observations) {
    // One timestep behind to match with prediction
    tf_obs.id = -1;
    
    // Distance between closest points between prediction and observation
    double max_distance = 1000;
    
    // For each prediction
    for (const LandmarkObs& sense : sense_range) {
      // Distance between prediction and observation
      double distance = dist(tf_obs.x, tf_obs.y, sense.x, sense.y);
      
      if (distance < max_distance) {
        max_distance = distance;
        tf_obs.id = sense.id;
      }
    }
  }
  //std::cout << "-----DATA ASSOCIATION DONE!-----" << std::endl; // DEBUG
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weight_normalizer = 0.0;
  for (Particle& particle : particles) {
    // Transformed observations
    vector<LandmarkObs> tf_observations;
    for (const LandmarkObs& observation : observations) {

      LandmarkObs tf_obs;
      coordinateTransformation(particle.x, particle.y, particle.theta, observation, tf_obs);
      tf_observations.push_back(tf_obs);
    }
    
    vector<LandmarkObs> sense_range;
    for (const Map::single_landmark_s& in_range : map_landmarks.landmark_list) {

      if (dist(particle.x, particle.y, in_range.x_f, in_range.y_f) < sensor_range) {
        sense_range.push_back(LandmarkObs{in_range.id_i, in_range.x_f, in_range.y_f});
      }
    }
    
    // dataAssociation to update observations with the landmark ID
    dataAssociation(tf_observations, sense_range);
    
    // Prepare data structures for storing associations for the single particle
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    
    // Reset particle weight
    particle.weight = 1.0;
    
    // calculate normalization term
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);
    
    double probability = 1.0;
    
    for (LandmarkObs& tf_obs : tf_observations) {
      // t for transformed observations
      double tx = tf_obs.x;
      double ty = tf_obs.y;
      double tid = tf_obs.id;

      for (LandmarkObs& in_range : sense_range) {
        // l for landmark observations
        double lx = in_range.x;
        double ly = in_range.y;
        double lid = in_range.id;
        
        if (tid == lid) {
          double x_diff2 = pow(tx - lx,2);
          double y_diff2 = pow(ty - ly,2);
          
          double sig_x2 = pow(sig_x,2);
          double sig_y2 = pow(sig_y,2);
          
          // Calculate the exponent
          double exponent = ((x_diff2)/(2 * sig_x2)) + ((y_diff2)/(2 * sig_y2));

          // Calculate the weight using normalization terms and exponent
          probability = gauss_norm * exp(-exponent);
          particle.weight *= probability;
        }
      }
      // Associations and observations
      associations.push_back(tid);
      sense_x.push_back(tx);
      sense_y.push_back(ty);
      
      SetAssociations(particle, associations, sense_x, sense_y);
    }  
    weight_normalizer += particle.weight;
  }
  // Normalize the probability
  for (int i=0; i < num_particles; i++) {
    particles[i].id = i;
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
  //std::cout << "-----WEIGHTS UPDATE DONE!-----" << std::endl; // DEBUG
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Random
  default_random_engine gen;
  
  // Discrete distribution with pmf equal to weights
  discrete_distribution<int> weights_pmf(weights.begin(), weights.end());
  // Initialize new particle array
  vector<Particle> resampled_particles;
  
  // Resample
  for (int i=0; i < num_particles; i++) {
    int x = weights_pmf(gen);
    resampled_particles.push_back(particles[x]);
  }  
  // Resampled becomes previous
  particles = resampled_particles;
  //std::cout << "-----RESAMPLE DONE!-----" << std::endl; // DEBUG
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}