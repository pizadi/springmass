// #include <GL/freeglut_std.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <ctime>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#define PI 3.14159
#define N 7
#define M 11
#define DAMPENING 1e-4

using namespace Eigen;
using namespace std;

// std::vector<Body> bodies;
// std::vector<Spring> springs;
VectorXd xpre;
MatrixXd A, f, Ai, colors;
auto t = std::chrono::high_resolution_clock::now();
MatrixXi connect = MatrixXi(M, 2);
VectorXd lrest = VectorXd::Zero(M), k = VectorXd::Zero(M), y = VectorXd(4*N), mass = VectorXd::Zero(N), dof = VectorXd::Ones(2*N);

Vector3f ccomp = Vector3f(1., 0., 0.), cstr = Vector3f(0., 0., 1.), cntr = Vector3f(1., 1., 1.);


void circle(VectorXd color, double x, double y, double size) {
    glColor3d(color(0), color(1), color(2));
    glBegin(GL_POLYGON);
    for (int i = 0; i < 12; i++) {
        glVertex2d(x + size * cos(PI * i / 6), y + size * sin(PI * i / 6));
    }
    glEnd();
}

void line(Vector2d a, Vector2d b) {
    glBegin(GL_LINES);
    glVertex2d(a(0), a(1));
    glVertex2d(b(0), b(1));
    glEnd();
}

void spring(int i) {
    Vector2d d1 = y.block(2*connect(i,1), 0, 2, 1),
             d0 = y.block(2*connect(i,0), 0, 2, 1);
    double diff = lrest(i) - (d1 - d0).norm();
    Vector3f color;
    double alpha = 1 / (1 + pow(diff * 10, 2));
    if (diff > 0) {
        color = alpha * cntr + (1 - alpha) * ccomp; 
    }
    else {
        color = alpha * cntr + (1 - alpha) * cstr;
    }
    glLineWidth(5.);
    glColor3f(color(0), color(1), color(2));
    glBegin(GL_LINES);
    glVertex2d(d0(0), d0(1));
    glVertex2d(d1(0), d1(1));
    glEnd();
}

VectorXd gradient(VectorXd y) {
    VectorXd out = VectorXd::Zero(4 * N);
    out.block(0, 0, 2*N, 1) = y.block(2*N, 0, 2*N, 1).array() * dof.array();
    for (int i = 0; i < M; i++) {
        Vector2d dx = y.block(2*connect(i,1), 0, 2, 1) - y.block(2*connect(i,0), 0, 2, 1);
        Vector2d f = k(i) * (dx.norm() - lrest(i)) * dx.normalized();
        out.block(2*N + 2 * connect(i,0), 0, 2, 1) += f / mass(connect(i,0));
        out.block(2*N + 2 * connect(i,1), 0, 2, 1) += - f / mass(connect(i,1));
        // cout << "x1:\n" << y.block(2*connect(i,1), 0, 2, 1) << endl;
        // cout << "x0:\n" << y.block(2*connect(i,0), 0, 2, 1) << endl;
        // cout << "dx:\n" << dx << endl << "f:\n" << f << endl;
        // break;
    }
    out.block(2*N, 0, 2*N, 1) = out.block(2*N, 0, 2*N, 1).array() * dof.array();
    return out;
}

void init(){
    glClearColor(0., 0., 0., 1.);
    glColor3f(1., 0., 0.);
    srand(time(NULL));
    // building f
    connect <<  0, 1,
                0, 2,
                1, 2,
                1, 3,
                2, 3,
                2, 4,
                3, 4,
                3, 5,
                4, 5,
                4, 6,
                5, 6;

    k.array() = 1e2;

    mass.array() = 1;

    dof(0) = dof(1) = 0.;
    dof(6) = dof(7) = 0.;

    y.block(0, 0, 2*N, 1) << .1, -.1,
                             .1, .2,
                             -.2, -.1,
                             -.2, .2,
                             -.4, .2,
                             -.35, .3,
                             -.6, .25;

    y.block(2*N,0,2*N,1) = 0. * y.block(2*N,0,2*N,1);

    for (int i = 0; i < M; i++) {
        lrest(i) = (y.block(2*connect(i,0), 0, 2, 1) - y.block(2*connect(i,1), 0, 2, 1)).norm() * .95;
    }

    // Ai = A.inverse();
    xpre = VectorXd::Zero(2*N);
    colors = MatrixXd::Random(N, 3).array() * .3 + .7;
    t = std::chrono::high_resolution_clock::now();
}

void draw() {
    glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < M; i++) {
        spring(i);
    }

    for (int i = 0; i < N; i++) {
        circle(colors.row(i), y(2*i), y(2*i+1), sqrt(mass(i)) * 0.03);
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

    y.block(2*N, 0, 2*N, 1) = exp(- dt * DAMPENING) * y.block(2*N, 0, 2*N, 1);
    
    // forward euler
    // y = y + dt * gradient(y);

    // backward euler
    // VectorXd yf = y + dt * gradient(y);
    // y = y + dt * gradient(yf);

    glutPostRedisplay();
}

void keyboard_cb(unsigned char key, int x, int y) {
    switch (key) {
        case 'q':
        case 'Q':
        exit(0);
        break;
    }
}

int main(int argc, char * argv[]){
    glutInit(&argc, argv);
    glutInitWindowSize(600, 600);
    glutInitWindowPosition(50, 50);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutCreateWindow("Test");

    glutDisplayFunc(draw);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard_cb);


    init();

    glutMainLoop();

    return 0;
}