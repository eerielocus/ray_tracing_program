#include "ofApp.h"

/*
Modified by Michael Kang for CS116A Project 2: Ray-tracing.
Modified by Michael Kang for CS116B Project 2: Ray-marching.
	- Recent modifications are tagged with CS116B
*/

// CS116B: Signed-distance function for scene.
float ofApp::sceneSDF(glm::vec3 point, SceneObject **object) {
	float closest = INFINITY;
	for (SceneObject *obj : scene) {
		float dist = obj->sdf(point);
		if (dist < closest) { 
			closest = dist;
			*object = obj;
		}
	}
	return closest;
}

// CS116B: Ray-march normal getter.
glm::vec3 ofApp::getNormalRM(glm::vec3 &p) {
	SceneObject *obj = NULL;
	float eps = 0.01; 
	float dp = sceneSDF(p, &obj); 
	glm::vec3 n(dp - sceneSDF(glm::vec3(p.x - eps, p.y, p.z), &obj), 
		dp - sceneSDF(glm::vec3(p.x, p.y - eps, p.z), &obj), 
		dp - sceneSDF(glm::vec3(p.x, p.y, p.z - eps), &obj)); 
	return glm::normalize(n);
}

// CS116B: Ray-marcher function and loop.
bool ofApp::raymarcher(Ray r, glm::vec3 &p, SceneObject **obj) {
	bool hit = false;
	p = r.p;
	for (int i = 0; i < 1500; i++) {
		float dist = sceneSDF(p, obj);
		if (dist < 0.01) {
			hit = true;
			break;
		}
		else if (dist > 100) { break; }
		else { p = p + r.d * dist; }
	}
	return hit;
}

// CS116B: Ray-march loop.
void ofApp::raymarch() {
	for (int j = 0; j < imageHeight; j++) {
		for (int i = 0; i < imageWidth; i++) {
			// Convert x, y to u, v.
			// Set distance to infinity and hit to false.
			float u = (i + 0.5) / imageWidth;
			float v = (j + 0.5) / imageHeight;
			bool hit = false;

			// Get ray projected from renderCam and run through list of scene objects
			// to determine if there is an object, and whether its the closest.
			Ray ray = renderCam.getRay(u, v);
			SceneObject *closestObj = NULL;
			glm::vec3 intersectNormal;
			glm::vec3 intersectPt;
			glm::vec3 closestNormal;	// Store closest normal and point for shading use.
			glm::vec3 closestPt;

			// Run through each object in scene to determine closest.
			// If there is an intersection, check distance between renderCam and intersect point,
			// and if it is less than currently set distance, replace distance and store object,
			// flag hit as true.
			hit = raymarcher(ray, intersectPt, &closestObj);
			if (hit) {
				closestPt = intersectPt;
				closestNormal = getNormalRM(closestPt);
			}

			// If there was a hit, set pixel to the closest object's color.
			// If not, set pixel to background color.

			// Update: check to see if texture is loaded, if it is, use texture method
			// to apply color variable to shading calculations.
			ofColor ambient;
			ofColor lamb;
			ofColor phon;
			ofColor texColor, specColor, normColor = ofFloatColor(0.5, 0.5, 1.0);
			if (hit) {
				if (closestObj->diffuseTex.isAllocated()) {
					closestObj->texture(closestPt, texColor, specColor, normColor);
					ambient = (texColor) * 0.03;
					lamb = lambert(closestObj, closestPt, closestNormal, texColor, normColor);
					phon = phong(closestObj, closestPt, closestNormal, texColor, specColor, normColor, power);
				}
				else {
					ambient = (closestObj->diffuseColor) * 0.05;
					lamb = lambert(closestObj, closestPt, closestNormal, closestObj->diffuseColor, normColor);
					phon = phong(closestObj, closestPt, closestNormal, closestObj->diffuseColor, closestObj->specularColor, normColor, power);
				}

				image.setColor(i, (imageHeight - 1) - j, ambient + lamb + phon);
			}
			else { image.setColor(i, (imageHeight - 1) - j, ofGetBackgroundColor()); }

			/*
			for (Sphere *s : atmosphere) {
				if (s->intersect(ray, intersectPt, intersectNormal)) {
					ofColor refr;
					ofColor temp = image.getColor(i, (imageHeight - 1) - j);
					ofColor test = atmoScatter(s, ray, intersectPt, intersectNormal, temp);
					image.setColor(i, (imageHeight - 1) - j, test);
				}
			}
			*/
		}
	}
	// Update and save image.
	cout << "Ray-traced render complete." << endl;
	image.update();
	image.save("render.jpg");
}

// Plane methods:
// Intersect Ray with Plane.  (wrapper on glm::intersect*)
bool Plane::intersect(const Ray &ray, glm::vec3 &point, glm::vec3 &normalAtIntersect) {
	float dist;
	bool insidePlane = false;
	bool hit = glm::intersectRayPlane(ray.p, ray.d, position, this->normal, dist);
	if (hit) {
		Ray r = ray;
		point = r.evalPoint(dist);
		normalAtIntersect = this->normal;
		// Modified to work with different facing normals.
		glm::vec3 objSpace = glm::inverse(getMatrix()) * glm::vec4(point, 1.0);
		float tempU = glm::dot(u, objSpace);
		float tempV = glm::dot(v, objSpace);
		if (tempU < max.x && tempU > min.x && tempV < max.y && tempV > min.y) { insidePlane = true; }
	}
	return insidePlane;
}

// Set UV coordinates and min/max boundaries of plane.
void Plane::setCoordinates() {
	// Get normalized direction vectors for surface of plane.=
	u = glm::vec3(1, 0, 0);
	v = glm::vec3(0, 1, 0);
	min = glm::vec2(-(width / 2), -(height / 2));
	max = glm::vec2((width / 2), (height / 2));
	maxD = glm::length(max);
}

// Apply texture to surface of plane.
void Plane::texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap) {
	// Use dot product to find how far along the uv the point is,
	// then divide by the size of the plane. Use scale to adjust number
	// of repeats and then find the texel value.
	glm::vec3 objSpace = glm::inverse(getMatrix()) * glm::vec4(point, 1.0);
	float i = (((glm::dot(u, objSpace) - min.x) / (max.x - min.x)) * repeat * diffuseTex.getWidth()) - 0.5;
	float j = (((glm::dot(v, objSpace) - min.y) / (max.y - min.y)) * repeat * diffuseTex.getHeight()) - 0.5;
	// Get the color from the texture using remainder function to repeat
	// if either i or j is out of bounds.
	diff = diffuseTex.getColor(fmod(i, diffuseTex.getWidth()), fmod(j, diffuseTex.getHeight()));
	// Check if spec map is added, if not, use spec color.
	if (specularTex.isAllocated()) { spec = specularTex.getColor(fmod(i, specularTex.getWidth()), fmod(j, specularTex.getHeight())); }
	else { spec = specularColor; }
}

// Sphere methods:
// Apply texture to sphere using sphere mapping.
void Sphere::texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap) {
	// Get local point on sphere.
	glm::vec3 normal = point - position;
	glm::normalize(normal);
	// Use parametric and modify to work with texture coordinates.
	float i = (0.6 - atan2(normal.z, normal.x) / (2 * PI)) * diffuseTex.getWidth();
	float j = (0.5 - asin(normal.y / radius) / PI) * diffuseTex.getHeight();
	// Apply same method of getting color as plane using fmod for repeats.
	diff = diffuseTex.getColor(fmod(i, diffuseTex.getWidth()), fmod(j, diffuseTex.getHeight()));

	if (specularTex.isAllocated()) { spec = specularTex.getColor(fmod(i, specularTex.getWidth()), fmod(j, specularTex.getHeight())); }
	else { spec = specularColor; }

	if (normalTex.isAllocated()) { normap = normalTex.getColor(fmod(i, normalTex.getWidth()), fmod(j, normalTex.getHeight())); }
}

// Cube methods:
// Determine intersection using Slab algorithm which uses bounding box values
// and compares each to find the min/max.
bool Cube::intersect(const Ray &ray, glm::vec3 &point, glm::vec3 &normal) {
	// Get ray in object space.
	glm::vec4 p = glm::inverse(getMatrix()) * glm::vec4(ray.p.x, ray.p.y, ray.p.z, 1.0);
	glm::vec4 p1 = glm::inverse(getMatrix()) * glm::vec4(ray.p + ray.d, 1.0);
	glm::vec3 d = glm::normalize(p1 - p);
	// Get inverted direction of ray.
	Ray r = Ray(p, d);
	glm::vec3 invDir = 1.0f / r.d;

	double t1 = (min[0] - r.p[0])*invDir[0];
	double t2 = (max[0] - r.p[0])*invDir[0];
	double tmin = glm::min(t1, t2);
	double tmax = glm::max(t1, t2);

	// Run through each coordinate and determine min/max.
	for (int i = 1; i < 3; i++) {
		t1 = (min[i] - r.p[i])*invDir[i];
		t2 = (max[i] - r.p[i])*invDir[i];
		tmin = glm::max(tmin, glm::min(glm::min(t1, t2), tmax));
		tmax = glm::min(tmax, glm::max(glm::max(t1, t2), tmin));
	}

	// If it is a hit:
	if (tmax > glm::max(tmin, 0.0)) {
		// Determine point of intersection.
		point = getMatrix() * glm::vec4(r.evalPoint(tmin), 1.0);
		
		// The idea here is to get the point of intersection in object space which should be on one
		// of the surfaces of the cube. By simply turning the float point into an integer (not rounding) 
		// we can determine on which side the point is on the unit cube. Since the cube will always be
		// the same 1x1x1 size unit cube and any scaling is done by matrix transform.
		// 
		// The bias is simply an arbitrary number to avoid weird shadowing dots across the surface caused
		// by the casting to int being 'not quite 1'.
		glm::vec3 hit = r.evalPoint(tmin);
		float bias = 1.000001;

		// Normalize and return point in world space.
		// The vec4's 'w' is set to 0 as I found out that for points, you want w = 1, but for directions, you want w = 0.
		normal = glm::normalize(getMatrix() * glm::vec4(float(int((hit.x) * bias)), float(int((hit.y) * bias)), float(int((hit.z) * bias)), 0.0));
		return true;
	}
	// Not a hit.
	return false;
}

// Texture cube using similar method used for planes.
void Cube::texture(glm::vec3 &point, ofColor &diff, ofColor &spec, ofColor &normap) {
	float minW = min.x, maxW = max.x, minH = min.y, maxH = max.y;

	// Temporary method to obtain UV directions.
	glm::vec3 hit = glm::inverse(getMatrix()) * glm::vec4(point, 1.0);
	if (abs(hit.x) > abs(hit.y) && abs(hit.x) > abs(hit.z)) {
		u = glm::vec3(0, 0, 1);
		v = glm::vec3(0, 1, 0);
	}
	else if (abs(hit.y) > abs(hit.x) && abs(hit.y) > abs(hit.z)) {
		u = glm::vec3(0, 0, 1);
		v = glm::vec3(1, 0, 0);
	}
	else if (abs(hit.z) > abs(hit.x) && abs(hit.z) > abs(hit.y)) {
		u = glm::vec3(1, 0, 0);
		v = glm::vec3(0, 1, 0);
	}

	// With local UV and bounding, utilize same method as plane texturing.
	float i = (((glm::dot(u, hit) - minW) / (maxW - minW)) * repeat * diffuseTex.getWidth()) - 0.5;
	float j = (((glm::dot(v, hit) - minH) / (maxH - minH)) * repeat * diffuseTex.getHeight()) - 0.5;
	// Get the color from the texture using remainder function to repeat
	// if either i or j is out of bounds.
	diff = diffuseTex.getColor(fmod(i, diffuseTex.getWidth()), fmod(j, diffuseTex.getHeight()));
	// Check if spec map is added, if not, use spec color.
	if (specularTex.isAllocated()) { spec = specularTex.getColor(fmod(i, specularTex.getWidth()), fmod(j, specularTex.getHeight())); }
	else { spec = specularColor; }
}

// Convert (u, v) to (x, y, z).
// We assume u,v is in [0, 1].
//
glm::vec3 ViewPlane::toWorld(float u, float v) {
	float w = width();
	float h = height();
	return (glm::vec3((u * w) + min.x, (v * h) + min.y, position.z));
}

// Get a ray from the current camera position to the (u, v) position on
// the ViewPlane.
//
Ray RenderCam::getRay(float u, float v) {
	glm::vec3 pointOnPlane = view.toWorld(u, v);
	return(Ray(position, glm::normalize(pointOnPlane - position)));
}

void RenderCam::drawFrustum() {
	view.draw();
	Ray r1 = getRay(0, 0);
	Ray r2 = getRay(0, 1);
	Ray r3 = getRay(1, 1);
	Ray r4 = getRay(1, 0);
	float dist = glm::length((view.toWorld(0, 0) - position));
	r1.draw(dist);
	r2.draw(dist);
	r3.draw(dist);
	r4.draw(dist);
}

// Ray tracing algorithm utilizing intersectRaySphere and rays projecting from renderCam.
void ofApp::raytrace() {
	for (int j = 0; j < imageHeight; j++) {
		for (int i = 0; i < imageWidth; i++) {
			// Convert x, y to u, v.
			// Set distance to infinity and hit to false.
			float u = (i + 0.5) / imageWidth;
			float v = (j + 0.5) / imageHeight;
			float distance = INFINITY;
			bool hit = false;

			// Get ray projected from renderCam and run through list of scene objects
			// to determine if there is an object, and whether its the closest.
			Ray ray = renderCam.getRay(u, v);
			SceneObject *closestObj = NULL;
			glm::vec3 intersectNormal;
			glm::vec3 intersectPt;
			glm::vec3 closestNormal;	// Store closest normal and point for shading use.
			glm::vec3 closestPt;

			// Run through each object in scene to determine closest.
			// If there is an intersection, check distance between renderCam and intersect point,
			// and if it is less than currently set distance, replace distance and store object,
			// flag hit as true.
			for (SceneObject *obj : scene) {
				if (obj->intersect(ray, intersectPt, intersectNormal)) {
					if (glm::distance(intersectPt, renderCam.position) < distance) { 
						distance = glm::distance(intersectPt, renderCam.position);
						closestPt = intersectPt;
						closestNormal = intersectNormal;
						closestObj = obj;
						hit = true;
					}
				}
			}
			
			// If there was a hit, set pixel to the closest object's color.
			// If not, set pixel to background color.

			// Update: check to see if texture is loaded, if it is, use texture method
			// to apply color variable to shading calculations.
			ofColor ambient;
			ofColor lamb;
			ofColor phon;
			ofColor texColor, specColor, normColor = ofFloatColor(0.5, 0.5, 1.0);
			if (hit) { 
				if (closestObj->diffuseTex.isAllocated()) {
					closestObj->texture(closestPt, texColor, specColor, normColor);
					ambient = (texColor) * 0.03;
					lamb = lambert(closestObj, closestPt, closestNormal, texColor, normColor);
					phon = phong(closestObj, closestPt, closestNormal, texColor, specColor, normColor, power);
				}
				else { 
					ambient = (closestObj->diffuseColor) * 0.05;
					lamb = lambert(closestObj, closestPt, closestNormal, closestObj->diffuseColor, normColor);
					phon = phong(closestObj, closestPt, closestNormal, closestObj->diffuseColor, closestObj->specularColor, normColor, power);
				}

				image.setColor(i, (imageHeight - 1) - j, ambient + lamb + phon);
			}
			else { image.setColor(i, (imageHeight - 1) - j, ofGetBackgroundColor()); }

			for (Sphere *s : atmosphere) {
				if (s->intersect(ray, intersectPt, intersectNormal)) {
					ofColor refr;
					ofColor temp = image.getColor(i, (imageHeight - 1) - j);
					ofColor test = atmoScatter(s, ray, intersectPt, intersectNormal, temp);
					/*
					if (hit) { refr = atmoRefraction(s, intersectPt, intersectNormal, texColor, specColor, temp, power); } 
					else { refr = atmoRefraction(s, intersectPt, intersectNormal, s->diffuseColor, s->specularColor, temp, power); }
					*/
					image.setColor(i, (imageHeight - 1) - j, test);
				}
			}
		}
	}
	// Update and save image.
	cout << "Ray-traced render complete." << endl;
	image.update();
	image.save("render.jpg");
}

// Normal mapping utilizing tangent space to calculate the necessary directions to adjust
// normal vector of point on object using information from the normal map's RGB.
// The default value is (0.5, 0.5, 1.0) which is 'straight up' that gives a bluish color
// to all normal maps.
//
// Tangent space concepts and conversion equation for normal map RGB taken from: https://learnopengl.com/Advanced-Lighting/Normal-Mapping
void ofApp::normalMapping(SceneObject *obj, glm::vec3 &p, glm::vec3 &norm, ofColor normap) {
	// Setting up tangent space and TBN matrix.
	glm::vec3 mainAxis = obj->position + (glm::vec3(0, 1, 0) * obj->radius);
	glm::vec3 tangent = glm::normalize(glm::cross(mainAxis, (p - obj->position)));
	glm::vec3 bitangent = glm::cross(norm, tangent);
	glm::vec3 T = glm::normalize(glm::vec3(obj->getMatrix() * glm::vec4(tangent, 0)));
	glm::vec3 B = glm::normalize(glm::vec3(obj->getMatrix() * glm::vec4(bitangent, 0)));
	glm::vec3 N = glm::normalize(glm::vec3(obj->getMatrix() * glm::vec4(norm, 0)));
	glm::mat3 TBN = glm::mat3(T, B, N);
	// Take the normal map information, apply conversion and multiply to TBN matrix.
	glm::vec3 normapN = glm::normalize(glm::vec3(normap.r * 2.0 - 1.0, normap.g * 2.0 - 1.0, normap.b * 2.0 - 1.0));
	norm = glm::normalize(TBN * normapN);
}

// Lambert shading function.
ofColor ofApp::lambert(SceneObject *obj, glm::vec3 &p, glm::vec3 &norm, ofColor diffuse, ofColor normap) {
	// Set initial color to be black.
	ofColor lit = ofColor::black;
	SceneObject *closestObj = NULL;

	// Go through each light to determine whether the ray from the light source
	// and object collides with another, if it does, do not shade.
	// If not, shade using diffuse algorithm.
	for (Light *light : lights) {
		float radius = glm::distance(light->position, p);

		// Get normalized vector of light to intersect point.
		glm::vec3 l = glm::normalize(light->position - p);
		glm::vec3 n = norm;
		glm::vec3 intersectPt;
		if (obj->normalTex.isAllocated()) { normalMapping(obj, p, n, normap); }

		// Adjust start position of shadow ray to avoid noise.
		Ray shadow = Ray(p + (n * 0.01), l);

		// CS116B: Ray-march version for shadows.
		if (raymarcher(shadow, intersectPt, &closestObj)) {
			if (glm::distance(shadow.p, intersectPt) > radius) {
				lit += diffuse * (light->intensity / glm::pow2(radius)) * glm::max(0.0F, glm::dot(n, l));
			}
		}
		else { lit += diffuse * (light->intensity / glm::pow2(radius)) * glm::max(0.0F, glm::dot(n, l)); }

		// If no intersection and within light, continue with shading.
		/*
		if (!light->castShadow(scene, shadow, radius) && light->withinLight(l)) {
			lit += diffuse * (light->intensity / glm::pow2(radius)) * glm::max(0.0F, glm::dot(n, l));
		}
		*/
	}

	return lit;
}

// Phong shading function.
ofColor ofApp::phong(SceneObject *obj, glm::vec3 &p, glm::vec3 &norm, ofColor diffuse, ofColor specular, ofColor normap, float power) {
	ofColor lit = ofColor::black;

	// Get the normalized view vector (camera vision).
	glm::vec3 v = glm::normalize(renderCam.position - p);
	SceneObject *closestObj = NULL;

	// Run through lights to determine if shadows/shading necessary.
	for (Light *light : lights) {
		float radius = glm::distance(light->position, p);

		// Get normalize vector from light to intersect point and determine the half-vector based on
		// view and light.
		glm::vec3 l = glm::normalize(light->position - p);
		glm::vec3 h = glm::normalize((v + l) / glm::length(v + l));
		glm::vec3 n = norm;
		glm::vec3 intersectPt;
		if (obj->normalTex.isAllocated()) { normalMapping(obj, p, n, normap); }
		
		Ray shadow = Ray(p + (n * 0.01), l);

		// CS116B: Ray-march version for shadows.
		if (raymarcher(shadow, intersectPt, &closestObj)) {
			if (glm::distance(shadow.p, intersectPt) > radius) {
				lit += specular * (light->intensity / glm::pow2(radius)) * glm::pow(glm::max(0.0F, glm::dot(n, h)), power);
			}
		}
		else { lit += specular * (light->intensity / glm::pow2(radius)) * glm::pow(glm::max(0.0F, glm::dot(n, h)), power); }

		// If no intersection and within light, continue with shading.
		/*
		if (!light->castShadow(scene, shadow, radius) && light->withinLight(l)) {
			lit += specular * (light->intensity / glm::pow2(radius)) * glm::pow(glm::max(0.0F, glm::dot(n, h)), power);
		}
		*/
	}

	return lit;
}

// Improved version (from my initial attempt below) of atmospheric scattering/refraction.
// Uses Rayleigh scattering techniques with some constants and equations taken from
// a tutorial written at: https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky
// 
// There are a lot of improvements to performance and generalizations I would like to add,
// but was more focused on getting things to work properly and look decent.
//
ofColor ofApp::atmoScatter(Sphere *obj, Ray r, glm::vec3 &p, glm::vec3 &norm, ofColor base) {
	// Some initial variables.
	ofColor result, rim = base;
	bool hitGround = false;			// A check for ground contact for color.
	float aOut = 0, aIn = 0;		// aOut: distance inside atmosphere, aIn: distance from camera
	float radius = glm::distance(lights[0]->position, p);
	float intensity = lights[0]->intensity * 2;
	glm::vec3 l = glm::normalize(lights[0]->position - p);
	// Since the sun is a large light source, it is safe to presume that the light direction simply a single direction
	// for this exercise. 
	glm::vec3 sunDir = glm::normalize(lights[0]->position - obj->position);
	glm::vec3 intersectNormal;
	glm::vec3 atmos, ground;
	glm::vec3 rayleighSum = glm::vec3(0, 0, 0);
	// Constants taken from the Rayleigh equation.
	const glm::vec3 betaR(3.8e-6f, 13.5e-6f, 33.1e-6f);
	// Make a new ray with a slight offset to avoid noise (like shadow ray).
	Ray atmo = Ray(p + (r.d * 0.001), r.d);

	// Check if the ray hits the atmosphere on the other side,
	// if it hits the ground, determine if its at the edge or not,
	// if it is, continue to apply atmospheric scattering on to it
	// so that it can provide a smoother transition between atmosphere
	// and terrain.
	if (obj->intersect(atmo, atmos, intersectNormal)) {
		if (!scene[0]->intersect(atmo, ground, intersectNormal)) {
			aOut = glm::distance(p, atmos);
			aIn = glm::distance(r.p, p);
		}
		else {
			hitGround = true;
			Ray edge = Ray(ground + (r.d * 0.01), r.d);
			if (scene[0]->intersect(edge, ground, intersectNormal)) {
				if (glm::distance(edge.p, ground) < 0.5) {
					aOut = glm::distance(p, atmos);
					aIn = glm::distance(r.p, p);
				}
			}
		}
	}

	// Temporary fix to blackening of atmosphere when sun is directly behind planet.
	if (sunDir == renderCam.aim) { sunDir = -sunDir; }
	float step = aOut / 16, stepPos = 0;			// The size of each iteration through the ray march.
	float sunAngle = glm::dot(r.d, sunDir);			// The cosine angle of camera and sun direction.
	float g = 0.76 * 0.76;							// Another constant for Rayleigh equation.
	float rayleigh = (3.0 / (16 * PI)) * (1.0 + glm::pow2(sunAngle));	// The Rayleigh equation.
	float rayleighDepth = 0;

	// Iterate through number of samples, using ray march to traverse through the atmosphere,
	// taking into account height to calculate depth.
	for (int i = 0; i < 16; i++) {
		glm::vec3 sample = r.evalPoint(aIn + step * 0.5);
		float height = glm::length(sample) - 1.5;
		float hRayleigh = exp(-height / 0.5) * step;	// Determines the brightness based on height.
		rayleighDepth += hRayleigh;

		float rayleighLightD = 0;
		float lightStep = 0, lightPos = 0;
		bool out = false;
		Ray starlight = Ray(sample, sunDir);
		// Check to see if sun light is able to reach this point.
		if (obj->intersect(starlight, sample, intersectNormal)) {
			lightStep = glm::distance(starlight.p, sample) / 6;
			for (int j = 0; j < 6; j++) {
				glm::vec3 sampleLight = starlight.evalPoint(lightPos + lightStep * 0.5);
				float heightL = glm::length(sampleLight) - 1.5;
				if (heightL < 0) { out = true; }

				rayleighLightD += exp(-heightL / 0.5) * lightStep;
				lightPos += lightStep;
			}
			// If successfully iterate through samples, apply attenuation using calculated
			// Rayleigh optical depths.
			if (!out) {
				glm::vec3 tau = betaR * (rayleighDepth + rayleighLightD);
				glm::vec3 attenuation = glm::vec3(exp(-tau.x), exp(-tau.y), exp(-tau.z));
				rayleighSum += attenuation * hRayleigh;
			}
		}
		stepPos += step;
	}
	// Process into color, taking into account distance from sun, direction, and normal.
	glm::vec3 color = (rayleighSum * betaR * rayleigh);
	result = ofFloatColor((pow(color.x, 0.5)) * 255 * (intensity / (radius * 2)) * glm::max(0.0F, glm::dot(norm, sunDir)),
						  (pow(color.y, 0.5)) * 255 * (intensity / (radius * 2)) * glm::max(0.0F, glm::dot(norm, sunDir)),
						  (pow(color.z, 0.5)) * 255 * (intensity / (radius * 2)) * glm::max(0.0F, glm::dot(norm, sunDir)));
	result += rim;
	return result;
}

// Atmospheric refraction function.
// This is initial attempt at recreating Earth-like atmospheric refraction and scattering.
// This prototype, while achieving the effect of light bending around, produced a fairly poor image with no
// scattering of light within the atmosphere. There were a lot of iterations before this one to try and
// accomodate some volume in the atmosphere and color change, it is currently set to be mostly white since
// I was testing what happens when light source is behind planet.
//
ofColor ofApp::atmoRefraction(Sphere *a, glm::vec3 &p, glm::vec3 &norm, ofColor diffuse, ofColor spec, ofColor ref, float power) {
	ofColor lit = diffuse;
	// Attempt to 'bounce' the direction of ray based on normal.
	glm::vec3 t = glm::normalize(p - renderCam.position);
	glm::vec3 r = glm::normalize(t + (norm));
	glm::vec3 intersectNormal;
	glm::vec3 intersectPt;
	// Make new ray from atmosphere to see if its hits planet or atmosphere.
	Ray atmo = Ray(p, r);
	glm::vec3 refract = atmo.evalPoint(a->radius / 5);
	// Run through lights to see if there are any that are reached without hitting objects.
	for (Light *light : lights) {
		float radius = glm::distance(light->position, p);
		float dist = glm::distance(refract, a->position);
		glm::vec3 dir = glm::normalize(light->position - refract);
		glm::vec3 l = glm::normalize((-1 * light->position) - p);
		glm::vec3 v = glm::normalize(renderCam.position - p);
		glm::vec3 h = glm::normalize((v + l) / glm::length(v + l));
		Ray bounce = Ray(refract, dir);
		// If there is a planetary body hit, return base color.
		for (SceneObject *obj : scene) {
			if (obj->intersect(bounce, intersectPt, intersectNormal)) {
				if (glm::distance(intersectPt, light->position) < radius) {
					return ref + ofColor::black;
				}
			}
		}
		// Attempt to produce colors appropriately, WIP..
		atmo.d = t;
		if (a->intersect(atmo, intersectPt, intersectNormal)) {
			float temp_dist = glm::distance(intersectPt, p);
			for (SceneObject *obj : scene) {
				if (obj->intersect(atmo, intersectPt, intersectNormal)) {
					if (glm::distance(intersectPt, p) < temp_dist) {
						lit -= ofColor::black;
					}
					else {
						lit += ref * (light->intensity / glm::pow2(radius));
					}
				}
				else {
					lit += a->diffuseColor * (light->intensity / radius);
				}
			}
		}
		lit += ref * a->diffuseColor * (light->intensity / glm::pow2(radius)) * glm::max(0.0F, glm::dot(norm, l));
	}
	return lit;
}

//--------------------------------------------------------------
void ofApp::setup(){
	// Main EasyCam view setup.
	mainCam.setDistance(10.0);
	mainCam.lookAt(glm::vec3(0, 0, -1));

	// Preview cam view setup to match render view.
	previewCam.setPosition(glm::vec3(0, 0, 10));
	previewCam.lookAt(glm::vec3(0, 0, 1));
	previewCam.setNearClip(1);

	// Default to main cam. F1 for main cam, F2 for preview cam.
	theCam = &mainCam;
	ofSetBackgroundColor(ofColor::black);

	// GUI setup for intensity and power of Phong shader.
	gui.setup();
	gui.add(intensity.setup("Intensity", 15.0F, 1.0F, 30.0F));
	gui.add(power.setup("Power", 20.0F, 10.0F, 1000.0F));
	bHide = false;
	
	// Allocate image size and save to disk.
	image.allocate(imageWidth, imageHeight, OF_IMAGE_COLOR);
	image.save("render.jpg");

	// Load textures.
	wallTex.load("milkyway.jpg");
	wallSpec.load("milkywayspec.jpg");
	earthTex.load("earthcloud.jpg");
	earthSpec.load("earthcloudspec.jpg");
	earthNorm.load("cloudsnormal.jpg");
	borgTex.load("borg.jpg");
	borgSpec.load("borgspec.jpg");
	stone.load("stone.jpg");
	wood.load("wood.jpg");

	// Initialize lights and push into list of lights.
	light1 = Light(glm::vec3(-3, 6, 6), 25.0F);		// Sun light.
	light2 = Light(glm::vec3(2, 1, -7), 15.0F);		// Background light for stars.
	light3 = Light(glm::vec3(4, 4, 6), 15.0F);
	lights.push_back(&light1);
	lights.push_back(&light2);
	lights.push_back(&light3);

	// Initialize test subjects.
	wall1 = Plane(glm::vec3(0, -2, 0), glm::vec3(0, 1, 0), ofColor::grey, 20.0f, 20.0f);	// Floor.
	wall2 = Plane(glm::vec3(0, 3, -10), glm::vec3(0, 0, 1), ofColor::grey, 20.0f, 10.0f);	// Wall.
	wall3 = Plane(glm::vec3(-10, 3, 0), glm::vec3(1, 0, 0), ofColor::grey, 20.0f, 10.0f);	// Wall.
	wall4 = Plane(glm::vec3(5, 3, 0), glm::normalize(glm::vec3(-1, 1, 1)), ofColor::grey, 5.0f, 5.0f);	// Wall.
	wall5 = Plane(glm::vec3(0, 8, 0), glm::vec3(0, -1, 0), ofColor::grey, 20.0f, 20.0f);	// Ceiling.
	test1 = Sphere(glm::vec3(0, 0, -3), 2.0, ofColor::green);			// Earth
	//test2 = Sphere(glm::vec3(0, 0, 0), 2.05, ofColor::lightBlue);		// Atmosphere
	test3 = Torus(glm::vec3(3, -1.75, 3));
	test4 = Torus(glm::vec3(3.5, -1.7, 3.85), glm::vec3(45, 0, 0), glm::vec2(0.25, 0.1), ofColor::green);
	test5 = Cube(glm::vec3(-3, -1, 3), glm::vec3(2, 2, 2), glm::vec3(0, 45, 0));

	// Set textures.
	wall1.setTexture(wood, 5);
	wall2.setTexture(wallTex, 1);
	wall3.setTexture(stone, 5);
	wall4.setTexture(stone, 5);
	test1.setTexture(earthTex, 1);
	test5.setTexture(borgTex, 1);

	// Set specular.
	//wall1.setSpecular(wallSpec);
	wall2.setSpecular(wallSpec);
	//wall3.setSpecular(wallSpec);
	test1.setSpecular(earthSpec);
	//test1.normalTex = earthNorm;
	test5.setSpecular(borgSpec);

	// Push onto scene.
	//scene.push_back(&test1);
	//scene.push_back(&test3);
	//scene.push_back(&test4);
	//scene.push_back(&test5);
	scene.push_back(&wall1);
	scene.push_back(&wall2);
	scene.push_back(&wall3);
	scene.push_back(&wall4);
	scene.push_back(&wall5);

	//atmosphere.push_back(&test2);
}

//--------------------------------------------------------------
void ofApp::update(){
	// Adjust light intensity based on GUI adjustments.
	// Commented out for midterm purposes.
	/*
	for (Light *light : lights) {
		light->intensity = intensity;
	}
	*/
}

//--------------------------------------------------------------
void ofApp::draw(){
	// Draw GUI on key command h.
	if (bHide) { gui.draw(); }
	theCam->begin();

	// Draw scene objects.
	for (SceneObject *obj : scene) {
		ofSetColor(obj->diffuseColor);
		ofFill();
		obj->draw();
	}

	for (Light *lighter : lights) {
		ofSetColor(ofColor::white);
		ofFill();
		lighter->draw();
		lighter->drawDir();
	}

	theCam->end();

	// If raytraced when pressing 'r', show rendered image in center of screen.
	// Pressing 'v' will show/hide the image.
	if (bShowImage) { 
		image.setAnchorPoint(image.getWidth() / 2.0, image.getHeight() / 2.0);
		image.draw(ofGetWidth() / 2.0, ofGetHeight() / 2.0);
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	switch (key) {
	case OF_KEY_F1:
		theCam = &mainCam;
		bShowImage = false;
		break;
	case OF_KEY_F2:
		theCam = &previewCam;
		bShowImage = false;
		break;
	case 'r':
		raymarch();
		bShowImage = false;
		break;
	case 'R':
		raytrace();
		bShowImage = false;
		break;
	case 'v':
		if (bShowImage) { bShowImage = false; }
		else { bShowImage = true; }
		break;
	case 'h':
		if (bHide) { bHide = false; }
		else { bHide = true; }
		break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
