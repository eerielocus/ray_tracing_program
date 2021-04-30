#pragma once

#include "ofMain.h"
#include "glm/gtx/intersect.hpp"
#include "glm/gtx/euler_angles.hpp"
#include "ofxGui.h"

/*
Modified by Michael Kang for CS116A.
Modified by Michael Kang for CS116B:
	- Recent additions will have CS116B tag.
*/

//  General Purpose Ray class. 
//
class Ray {
public:
	Ray(glm::vec3 p, glm::vec3 d) { this->p = p; this->d = d; }
	void draw(float t) { ofDrawLine(p, p + t * d); }

	glm::vec3 evalPoint(float t) { return (p + t * d); }

	glm::vec3 p, d;
};

//  Base class for any renderable object in the scene.
//
class SceneObject {
public:
	virtual void draw() = 0;    // pure virtual funcs - must be overloaded
	virtual bool intersect(const Ray &ray, glm::vec3 &point, glm::vec3 &normal) { cout << "SceneObject::intersect" << endl; return false; }
	virtual void texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap) = 0;

	// CS116B addition: signed distance function.
	virtual float sdf(glm::vec3 &p) = 0;

	// Default texture setters.
	void setTexture(ofImage diff, float s = 1) { diffuseTex = diff; repeat = s; }
	void setSpecular(ofImage spec) { specularTex = spec; }

	// Utilize transform matrix for local-global conversions.
	glm::mat4 getMatrix() {
		glm::mat4 translate = glm::translate(glm::mat4(1.0), glm::vec3(position));
		glm::mat4 rotate = glm::eulerAngleYXZ(glm::radians(rotation.y), glm::radians(rotation.x), glm::radians(rotation.z));
		glm::mat4 sc = glm::scale(glm::mat4(1.0), glm::vec3(scale.x, scale.y, scale.z));
		return translate * rotate * sc;
	}

	// Any data common to all scene objects goes here:
	glm::vec3 position = glm::vec3(0, 0, 0);
	glm::vec3 rotation = glm::vec3(0, 0, 0);
	glm::vec3 scale = glm::vec3(1, 1, 1);
	glm::vec3 u, v;
	float repeat; // Temporary var for adjusting texture repeats.
	float radius = 0;

	// Material properties (we will ultimately replace this with a Material class - TBD)
	//
	ofColor diffuseColor = ofColor::grey;    // Default colors - can be changed.
	ofColor specularColor = ofColor::lightGray;
	ofImage diffuseTex;
	ofImage specularTex;
	ofImage normalTex;
};

class Light : public SceneObject {
public:
	Light(glm::vec3 p, float i = 1.0) { position = p; intensity = i; }
	Light() {}

	void draw() { ofDrawSphere(position, 0.1); }
	// Draw direction of light as a line, for point lights, do nothing.
	virtual void drawDir() {}

	// Check if the object is within the light, for point lights, always return true.
	virtual bool withinLight(glm::vec3 l) { return true; }
	void texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap) {}
	float sdf(glm::vec3 &p) { return 0; }

	// Will most likely move back to ofApp.cpp - or to separate class file.
	// Check if objects in scene is receiving shadows.
	// Pass list of objects in scene, the shadow ray, and distance from light.
	bool castShadow(vector<SceneObject *> scene, Ray shadow, float radius) {
		glm::vec3 intersectPt;
		glm::vec3 intersectNo;
		SceneObject *closestObj = NULL;
		bool cast = false;

		// Run through each object to check whether shadow ray intersects to anything.
		for (SceneObject *objs : scene) {
			if (objs->intersect(shadow, intersectPt, intersectNo)) {
				if (glm::distance(shadow.p, intersectPt) < radius) {
					cast = true;
				}
			}
		}
		return cast;
	}

	// Data:
	float intensity;
};

class Spotlight : public Light {
public:
	// Spotlight class derived from Light class.
	// Possible change in future:
	// Bring spotlight specific variables into this class and use a
	// cast on pointer to access variables in the base class pointer loop.
	Spotlight(glm::vec3 p, glm::vec3 d, float i = 1.0, float c = 15.0F) { 
		position = p; 
		direction = d - p;
		intensity = i; 
		cone = c;
	}
	Spotlight() {}

	// Draw direction as a line.
	void drawDir() { ofDrawLine(position, glm::normalize(direction)); }
	void texture(glm::vec3 &point) {}
	bool withinLight(glm::vec3 l) {
		// Determine whether the light vector is within the cone by utilizing dot product to find
		// cosine between light vector 'l' and the spotlight direction.
		// The direction of light 'l' needs to be reversed since it was initially calculated to go
		// from intersect point to light source for shadow calculations.
		// If the cosine angle between the light and spotlight direction is less than the cosine of
		// the cone angle, then skip to next light since it is outside of the cone.
			// cos(0) = 1, therefore any angle less than the cos(cone angle) is outside of it.
		float withinSpot = glm::dot(-l, glm::normalize(direction));
		if (withinSpot < glm::cos(glm::radians(cone))) { return false; }
		else { return true; }
	}

	// Data:
	// Direction of light for spotlights.
	glm::vec3 direction;
	// Cone radius for spotlights.
	float cone;
};

//  General purpose sphere.  (assume parametric)
//
class Sphere : public SceneObject {
public:
	Sphere(glm::vec3 p, float r, ofColor diffuse = ofColor::lightGray) { position = p; radius = r; diffuseColor = diffuse; }
	Sphere() {}

	bool intersect(const Ray &ray, glm::vec3 &point, glm::vec3 &normal) {
		return (glm::intersectRaySphere(ray.p, ray.d, position, radius, point, normal));
	}

	// Signed-distance function:
	float sdf(glm::vec3 &p) { return glm::length(position - p) - radius; }

	void draw() { ofDrawSphere(position, radius); }
	void texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap);
};

//  Mesh class (will complete later- this will be a refinement of Mesh from Project 1).
//
class Mesh : public SceneObject {
	bool intersect(const Ray &ray, glm::vec3 &point, glm::vec3 &normal) { return false; }
	void draw() { }
};

// Cube class.
//
class Cube : public SceneObject {
public:
	Cube(ofColor color = ofColor::white) { diffuseColor = color; }
	Cube(glm::vec3 p, glm::vec3 d = glm::vec3(2.0, 2.0, 2.0), glm::vec3 r = glm::vec3(0, 0, 0), ofColor color = ofColor::white) {
		position = p;
		rotation = r;
		dimension = d;
		diffuseColor = color;
		setCoordinates();
	}

	bool intersect(const Ray &ray, glm::vec3 &point, glm::vec3 &normal);
	void texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap);
	void setCoordinates() {
		min = glm::vec3(-(dimension.x / 2), -(dimension.y / 2), -(dimension.z / 2));
		max = glm::vec3((dimension.x / 2), (dimension.y / 2), (dimension.z / 2));
	}

	// CS116B: Cube ray-march SDF.
	float sdf(glm::vec3 &p) { 
		glm::vec3 o = glm::inverse(getMatrix()) * glm::vec4(p, 1);
		glm::vec3 q = abs(o) - (dimension * 0.5);
		return glm::length(glm::max(q, glm::vec3(0, 0, 0))) + glm::min(glm::compMax(q), 0.0F);
	}

	void draw() {
		ofPushMatrix();
		ofMultMatrix(getMatrix());
		ofNoFill();
		ofDrawBox(dimension.x, dimension.y, dimension.z);
		ofPopMatrix();
	}

	// Data:
	glm::vec3 min, max;
	glm::vec3 dimension = glm::vec3(2.0, 2.0, 2.0);
};

// CS116B: Torus implementation.
//
class Torus : public SceneObject {
public:
	Torus(glm::vec3 p, glm::vec3 r = glm::vec3(0, 0, 0), glm::vec2 t = glm::vec2(0.5, 0.25), ofColor c = ofColor::blue) {
		position = p;
		rotation = r;
		tParam = t;
		diffuseColor = c;
	}
	Torus() {}

	void draw() {}
	bool intersect(const Ray &ray, glm::vec3 &point, glm::vec3 &normal) { return false; }
	void texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap) {}

	float sdf(glm::vec3 &p) {
		glm::vec3 o = glm::inverse(getMatrix()) * glm::vec4(p, 1);
		glm::vec2 q = glm::vec2(glm::length(glm::vec2(o.x, o.z)) - tParam.x, o.y);
		return glm::length(q) - tParam.y;
	}

	glm::vec2 tParam = glm::vec2(0.5, 0.25);
};

// General purpose plane.
//
class Plane: public SceneObject {
public:
	Plane(glm::vec3 p, glm::vec3 n, ofColor diffuse = ofColor::darkOliveGreen, float w = 20, float h = 20) {
		position = p; normal = n;
		width = w;
		height = h;
		diffuseColor = diffuse;
		// Generalized rotation based on normal.
		// Check if normal is parallel to default, if it is, do not rotate.
		// Rotation uses cross product of input normal and default normal by 90 degrees.
		if (n == glm::vec3(0, 0, 1)) rotation = glm::vec3(0, 0, 0);
		else { rotation = glm::normalize(glm::cross(n, glm::vec3(0, 0, 1))) * 90; }
		setCoordinates();
	}
	Plane() { normal = glm::vec3(0, 1, 0); }

	bool intersect(const Ray &ray, glm::vec3 &point, glm::vec3 &normal);
	void setCoordinates();

	// CS116B: Finite plane SDF:
	// Utilized similar method to ray-trace finite plane, using transform matrix and dot product.
	float sdf(glm::vec3 &p) {
		float dist = glm::dot(normal, p - position);
		if (dist <= maxD) {
			glm::vec3 o = glm::inverse(getMatrix()) * glm::vec4(p, 1.0);
			float tempU = glm::dot(u, o);
			float tempV = glm::dot(v, o);
			if (tempU <= max.x && tempU >= min.x && tempV <= max.y && tempV >= min.y) {
				return dist;
			}
		}
		return dist + glm::length(p - position);
	}

	// Removed dependency on plane primitive and used simple OF draw for visual rep.
	void draw() {
		ofPushMatrix();
		ofMultMatrix(getMatrix()); 
		ofNoFill();
		ofDrawPlane(width, height);
		ofPopMatrix();
	}

	void texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap);
	glm::vec3 getNormal(const glm::vec3 &p) { return this->normal; }
	glm::vec3 normal;
	glm::vec2 min, max;

	// Data:
	float width = 20;
	float height = 20;
	float minW, maxW;	// Boundaries of plane.
	float minH, maxH;
	float maxD;
};

// View plane for render camera.
// 
class ViewPlane : public Plane {
public:
	ViewPlane(glm::vec2 p0, glm::vec2 p1) { min = p0; max = p1; }

	ViewPlane() {                         // Create reasonable defaults (6x4 aspect).
		min = glm::vec2(-8, -6);
		max = glm::vec2(8, 6);
		position = glm::vec3(0, 0, 2);
		normal = glm::vec3(0, 0, 1);      // Viewplane currently limited to Z axis orientation.
	}

	void setSize(glm::vec2 min, glm::vec2 max) { this->min = min; this->max = max; }
	float getAspect() { return width() / height(); }

	glm::vec3 toWorld(float u, float v);   //   (u, v) --> (x, y, z) [ world space ]

	void draw() {
		ofDrawRectangle(glm::vec3(min.x, min.y, position.z), width(), height());
	}


	float width() { return (max.x - min.x); }
	float height() { return (max.y - min.y); }
	float sdf(glm::vec3 &p) { return 0; }

	// Some convenience methods for returning the corners
	//
	glm::vec2 topLeft() { return glm::vec2(min.x, max.y); }
	glm::vec2 topRight() { return max; }
	glm::vec2 bottomLeft() { return min; }
	glm::vec2 bottomRight() { return glm::vec2(max.x, min.y); }

	//  To define an infinite plane, we just need a point and normal.
	//  The ViewPlane is a finite plane so we need to define the boundaries.
	//  We will define this in terms of min, max  in 2D.  
	//  (in local 2D space of the plane)
	//  ultimately, will want to locate the ViewPlane with RenderCam anywhere
	//  in the scene, so it is easier to define the View rectangle in a local'
	//  coordinate system.
	//
	glm::vec2 min, max;
};


//  Render camera  - currently must be z axis aligned (we will improve this in project 4)
//
class RenderCam : public SceneObject {
public:
	RenderCam() {
		position = glm::vec3(0, 0, 10);
		aim = glm::vec3(0, 0, -1);
	}

	Ray getRay(float u, float v);
	void draw() { ofDrawBox(position, 1.0); };
	void drawFrustum();
	void texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap) {}
	float sdf(glm::vec3 &p) { return 0; }

	glm::vec3 aim;
	ViewPlane view;          // The camera viewplane, this is the view that we will render.
};

class ofApp : public ofBaseApp{
	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
		bool bHide = true;
		bool bShowImage = false;

		// New ray-march method for CS116B:
		void raymarch();
		bool raymarcher(Ray r, glm::vec3 &p, SceneObject **obj);
		float sceneSDF(glm::vec3 p, SceneObject **obj);
		glm::vec3 getNormalRM(glm::vec3 &p);

		// Method to render using raytrace algorithm.
		void raytrace();
		void normalMapping(SceneObject *obj, glm::vec3 &p, glm::vec3 &norm, ofColor normap);

		// Shading methods.
		ofColor lambert(SceneObject *obj, glm::vec3 &p, glm::vec3 &norm, ofColor diffuse, ofColor normap);
		ofColor phong(SceneObject *obj, glm::vec3 &p, glm::vec3 &norm, ofColor diffuse, ofColor specular, ofColor normap, float power);
		ofColor atmoRefraction(Sphere *obj, glm::vec3 &p, glm::vec3 &norm, ofColor diffuse, ofColor base, ofColor ref, float index);
		ofColor atmoScatter(Sphere *obj, Ray r, glm::vec3 &p, glm::vec3 &norm, ofColor base);

		ofxPanel gui;
		ofxFloatSlider intensity;
		ofxFloatSlider power;

		ofEasyCam mainCam;
		ofCamera sideCam;
		ofCamera previewCam;
		ofCamera *theCam;    // Set to current camera either mainCam or sideCam.

		// Set up one render camera to render image through.
		RenderCam renderCam;
		ofImage image;

		// Textures:
		ofImage wallTex;
		ofImage wallSpec;
		ofImage earthTex;
		ofImage earthSpec;
		ofImage earthNorm;
		ofImage borgTex;
		ofImage borgSpec;
		ofImage stone;
		ofImage wood;

		// Scene objects list and test objects.
		vector<SceneObject *> scene;
		vector<Sphere *> atmosphere;
		Plane wall1;
		Plane wall2;
		Plane wall3;
		Plane wall4;
		Plane wall5;

		Sphere test1;
		Sphere test2;
		Torus test3;
		Torus test4;
		Cube test5;

		vector<Light *> lights;
		Light light1;
		Light light2;
		Light light3;
		Spotlight spot1;
		Spotlight spot2;

		int imageWidth = 1280;
		int imageHeight = 960;
};
