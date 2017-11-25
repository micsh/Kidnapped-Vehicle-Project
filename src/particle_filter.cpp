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

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	normal_distribution<double> N_x_init(0, std[0]);
	normal_distribution<double> N_y_init(0, std[1]);
	normal_distribution<double> N_theta_init(0, std[2]);

	num_particles = 50;

	weights = vector<double>(num_particles);
	particles = vector<Particle>(num_particles);

	for (int i = 0; i < num_particles; ++i)
	{
		Particle p;
		p.id = i;
		p.x = x + N_x_init(gen);
		p.y = y +  N_y_init(gen);
		p.theta = theta + N_theta_init(gen);
		p.weight = 1.0;

		particles[i] = p;
		weights[i] = p.weight;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	double x0 = 0;
	double y0 = 0;
	double theta0 = 0;

	double x_new = 0;
	double y_new = 0;
	double theta_new = 0;

	for (int i = 0; i < num_particles; ++i)
	{
		x0 = particles[i].x;
		y0 = particles[i].y;
		theta0 = particles[i].theta;

		if (fabs(yaw_rate) < 1e-5)
		{
			x_new = x0 + velocity * delta_t * cos(theta0);
			y_new = y0 + velocity * delta_t * sin(theta0);
			theta_new = theta0;
		}
		else
		{
			x_new = x0 + (velocity / yaw_rate) * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
			y_new = y0 + (velocity / yaw_rate) * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
			theta_new = theta0 + yaw_rate * delta_t;
		}

		particles[i].x = x_new + N_x(gen);
		particles[i].y = y_new + N_y(gen);
		particles[i].theta = theta_new + N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
}

Particle mapObservationToMapCoordinates(LandmarkObs observation, Particle particle)
{
	double x = observation.x;
	double y = observation.y;

	double xt = particle.x;
	double yt = particle.y;
	double theta = particle.theta;

	Particle mapCoordinates;

	mapCoordinates.x = x * cos(theta) - y * sin(theta) + xt;
	mapCoordinates.y = x * sin(theta) + y * cos(theta) + yt;

	return mapCoordinates;
}

double distanceFromLandmark(Map::single_landmark_s land_mark, Particle map_coordinates)
{
	return dist(land_mark.x_f, land_mark.y_f, map_coordinates.x, map_coordinates.y);
}

Map::single_landmark_s findClosestLandmark(Map map_landmarks, Particle map_coordinates)
{
	Map::single_landmark_s closest_landmark = map_landmarks.landmark_list[0];
	double distance = distanceFromLandmark(map_landmarks.landmark_list[0], map_coordinates);

	for (int i = 1; i < map_landmarks.landmark_list.size(); ++i)
	{
		Map::single_landmark_s current_landmark = map_landmarks.landmark_list[i];
		double current_distance = distanceFromLandmark(current_landmark, map_coordinates);

		if (current_distance < distance)
		{
			distance = current_distance;
			closest_landmark = current_landmark;
		}
	}

	return closest_landmark;
}

double findObservationProbability(Map::single_landmark_s closest_landmark, Particle map_coordinates, double std_landmark[])
{
	double mew_x = closest_landmark.x_f;
	double mew_y = closest_landmark.y_f;

	double x = map_coordinates.x;
	double y = map_coordinates.y;

	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	return (1 / (2 * M_PI * sigma_x * sigma_y)) * pow(M_E, -(pow(x - mew_x, 2) / (2 * pow(sigma_x, 2)) + pow(y - mew_y, 2) / (2 * pow(sigma_y, 2))));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < particles.size(); ++i)
	{
		Particle particle = particles[i];
		double new_weight = 1;
		for (LandmarkObs observation : observations)
		{
			// Convert the observation as seen by the particle into map coordinates
			Particle map_coordinates = mapObservationToMapCoordinates(observation, particle);
			Map::single_landmark_s closest_landmark = findClosestLandmark(map_landmarks, map_coordinates);
			double observation_probability = findObservationProbability(closest_landmark, map_coordinates, std_landmark);
			new_weight *= observation_probability;
		}
		particle.weight = new_weight;
		weights[i] = new_weight;
	}
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<> distribution(weights.begin(), weights.end());

	vector<Particle> new_particles = vector<Particle>(num_particles);
	for (int i = 0; i < particles.size(); ++i)
	{
		int index = distribution(gen);
		new_particles[i] = particles[index];
		weights[i] = new_particles[i].weight;
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const vector<int> &associations,
										 const vector<double> &sense_x, const vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
