// #include <GL/freeglut_std.h>
#include <GL/freeglut_std.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <cmath>
#include <ctime>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#define PI 3.14159
#define N 5
#define M 9
#define DAMPENING 1e-5
#define CAMSPEED 1e-1
#define DIST 3

using namespace Eigen;
using namespace std;

// std::vector<Body> bodies;
// std::vector<Spring> springs;
VectorXd xpre;
MatrixXd A, f, Ai, colors;
auto t = std::chrono::high_resolution_clock::now();
MatrixXi connect = MatrixXi(M, 2);
VectorXd lrest = VectorXd::Zero(M), k = VectorXd::Zero(M), y = VectorXd(6*N), mass = VectorXd::Zero(N), dof = VectorXd::Ones(3*N);
Vector3f ccomp = Vector3f(1., 0., 0.), cstr = Vector3f(0., 0., 1.), cntr = Vector3f(1., 1., 1.);
MatrixXd QtS = MatrixXd::Zero(3 * M, 3 * N), StL = MatrixXd::Zero(M, 3 * M), LtS, StQ;

float theta = PI / 2, phi = 0;

void line(Vector3d a, Vector3d b) {
    glBegin(GL_LINES);
    glVertex2d(a(0), a(1));
    glVertex2d(b(0), b(1));
    glEnd();
}

void spring(int i) {
    Vector3d d1 = y.block(3*connect(i,1), 0, 3, 1),
             d0 = y.block(3*connect(i,0), 0, 3, 1);
    Vector3d dist = d1 - d0;
    double diff = lrest(i) - dist.norm();
    Vector3f color;
    double alpha = 1 / (1 + pow(diff * sqrt(k(i)), 2));
    if (diff > 0) {
        color = alpha * cntr + (1 - alpha) * ccomp; 
    }
    else {
        color = alpha * cntr + (1 - alpha) * cstr;
    }
    glLineWidth(5.);
    glColor3f(color(0), color(1), color(2));
    float sprphi = atan2(dist(0), dist(1));
    float sprtheta = atan2(dist(2), dist.block(0,0,2,1).norm());
    glPushMatrix();
    glTranslated(d0(0), d0(1), d0(2));
    glRotatef(-sprphi * 180 / PI, 0, 0, 1);
    glRotatef(sprtheta * 180 / PI, 1, 0, 0);
    glRotatef(-90., 1, 0, 0);
    glutSolidCylinder(.01, (d1 - d0).norm(), 12, 12);
    glPopMatrix();
}

VectorXd gradient(VectorXd y) {
    VectorXd out = VectorXd::Zero(6*N), diff, diffsq, lc, force;
    out.block(0, 0, 3*N, 1) = y.block(3*N, 0, 3*N, 1).array() * dof.array();
    diff = QtS * y.block(0, 0, 3*N, 1);
    diffsq = diff.array().square();
    lc = (StL * diffsq).array().sqrt() - lrest.array();
    force = lc.array() * k.array();
    force = (force.array() / ((StL * diffsq).array().sqrt()));
    force = LtS * force;
    force = force.array() * diff.array();
    out.block(3*N, 0, 3*N, 1) = - StQ * force;
    out.block(3*N, 0, 3*N, 1) = out.block(3*N, 0, 3*N, 1).array() * dof.array();
    return out;
}

void init(){
    glClearColor(0., 0., 0., 1.);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1,1,-1,1, 2, DIST + 2);
    glMatrixMode(GL_MODELVIEW);

    srand(time(NULL));
    // building f
    connect <<  0, 1,
                0, 2,
                0, 3,
                1, 2,
                2, 3,
                3, 1,
                1, 4,
                2, 4,
                3, 4;

    k.array() = 1e3;

    mass.array() = 1;

    y.block(0, 0, 3*N, 1) << 
                            0, .5, 0,
                            -.42, 0, -.25,
                            0, 0, .5,
                            .42, 0, -.25,
                            0, -.5, 0;

    y.block(3*N, 0, 3*N, 1) = VectorXd::Random(3 * N);
    y.block(3*N,0,3*N,1) = 1e-1 * y.block(3*N,0,3*N,1);

    for (int i = 0; i < M; i++) {
        lrest(i) = (y.block(3*connect(i,0), 0, 3, 1) - y.block(3*connect(i,1), 0, 3, 1)).norm() * .8;
    }
    
    for (int i = 0; i < M; i++) {
        QtS.block(3 * i, 3 * (connect(i, 1)), 3, 3) = Matrix3d::Identity();
        QtS.block(3 * i, 3 * (connect(i, 0)), 3, 3) = - Matrix3d::Identity();
    }

    for (int i = 0; i < M; i++) {
        StL.block(i, 3 * i, 1, 3).array() = 1;
    }

    LtS = StL.transpose();
    StQ = QtS.transpose();

    colors = MatrixXd::Random(N, 3).array() * .3 + .7;
    t = std::chrono::high_resolution_clock::now();
}

void draw() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 

    glLoadIdentity();
    gluLookAt(- DIST * sin(theta) * sin(phi), DIST * cos(theta), - DIST * sin(theta) * cos(phi),0,0, 5 * 0,0,1,0);

    for (int i = 0; i < M; i++) {
        spring(i);
    }

    // glLoadIdentity();
    // glTranslatef(0, 0, -5);
    // glutSolidSphere(1., 12, 12);

    for (int i = 0; i < N; i++) {
        // glLoadIdentity();
        glPushMatrix();
        glTranslatef(y(3*i), y(3*i+1), y(3*i+2));
        glColor3d(colors(3*i), colors(3*i+1), colors(3*i+2));
        glutSolidSphere(sqrt(mass(i)) * 0.05, 40, 40);
        glPopMatrix();
    }


    glutSwapBuffers();
}

void idle() {

    auto d = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t);
    t = std::chrono::high_resolution_clock::now();
    double dt = 1 * d.count() * 1e-6;
    // fourth-order
    VectorXd y1 = gradient(y);
    VectorXd y2 = gradient(y + dt / 2 * y1);
    VectorXd y3 = gradient(y + dt / 2 * y2);
    VectorXd y4 = gradient(y + dt * y3);
    y = y + dt / 6 * (y1 + 2 * y2 + 2 * y3 + y4);

    y.block(3*N, 0, 3*N, 1) = exp(- dt * DAMPENING) * y.block(3*N, 0, 3*N, 1);
    
    glutPostRedisplay();
}

void keyboard_cb(unsigned char key, int x, int y) {
    switch (key) {
        case 'q':
        case 'Q':
        glutLeaveMainLoop();
        break;
        case 'k':
        case 'K':
        theta += CAMSPEED;
        break;
        case 'i':
        case 'I':
        theta -= CAMSPEED;
        break;
        case 'j':
        case 'J':
        phi -= CAMSPEED;
        break;
        case 'l':
        case 'L':
        phi += CAMSPEED;
        break;
    }
    theta = theta>PI-1e-5?PI-1e-5:theta;
    theta = theta<1e-5?1e-5:theta;
    while (phi > 2 * PI) phi -= 2 * PI;
    while (phi < 0) phi += 2 * PI;
    char title[100];
    sprintf(title, "%f %f", theta, phi);
    glutSetWindowTitle(title);
}

void mouse_cb(int mx, int my) {
    static int last_mx;
    static int last_my;
    theta += (double) (my - last_my) / 10 * CAMSPEED;
    phi += (double) (mx - last_mx) / 10 * CAMSPEED;
    theta = theta>PI-1e-5?PI-1e-5:theta;
    theta = theta<1e-5?1e-5:theta;
    while (phi > 2 * PI) phi -= 2 * PI;
    while (phi < 0) phi += 2 * PI;
    char title[100];
    sprintf(title, "%f %f", theta, phi);
    glutSetWindowTitle(title);
    last_mx = mx;
    last_my = my;
}

int main(int argc, char * argv[]){
    glutInit(&argc, argv);
    glutInitWindowSize(600, 600);
    glutInitWindowPosition(50, 50);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    glutCreateWindow("Test");

    glutDisplayFunc(draw);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard_cb);
    // glutPassiveMotionFunc(mouse_cb);


    init();

    glutMainLoop();

    return 0;
}