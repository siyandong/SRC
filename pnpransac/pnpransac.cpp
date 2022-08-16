// one-to-many PnP-RANSAC implementation for SRC.
/* 
PnP-RANSAC implementation based on DSAC++
Code: https://github.com/vislearn/LessMore
Paper: https://arxiv.org/abs/1711.10228
*/  

/*
Copyright (c) 2016, TU Dresden
Copyright (c) 2017, Heidelberg University
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "pnpransac.h"
#include <fstream> // debug inliers.
#include <opencv2/calib3d.hpp> // try pose refinement.

std::vector<std::mt19937> ThreadRand::generators;
bool ThreadRand::initialised = false;

void ThreadRand::forceInit(unsigned seed)
{
    initialised = false;
    init(seed);
}

void ThreadRand::init(unsigned seed)
{
    #pragma omp critical
    {
	if(!initialised)
	{
	    unsigned nThreads = omp_get_max_threads();
	    
	    for(unsigned i = 0; i < nThreads; i++)
	    {    
		generators.push_back(std::mt19937());
		generators[i].seed(i+seed);
	    }

	    initialised = true;
	}    
    }
}

int ThreadRand::irand(int min, int max, int tid)
{
    std::uniform_int_distribution<int> dist(min, max);

    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;
    
    if(!initialised) init();
  
    return dist(ThreadRand::generators[threadID]);
}

double ThreadRand::drand(double min, double max, int tid)
{
    std::uniform_real_distribution<double> dist(min, max);
    
    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;

    if(!initialised) init();

    return dist(ThreadRand::generators[threadID]);
}

double ThreadRand::dgauss(double mean, double stdDev, int tid)
{
    std::normal_distribution<double> dist(mean, stdDev);
    
    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;

    if(!initialised) init();

    return dist(ThreadRand::generators[threadID]);
}

int irand(int incMin, int excMax, int tid)
{
    return ThreadRand::irand(incMin, excMax - 1, tid);
}

double drand(double incMin, double incMax,int tid)
{
    return ThreadRand::drand(incMin, incMax, tid);
}

int igauss(int mean, int stdDev, int tid)
{
    return (int) ThreadRand::dgauss(mean, stdDev, tid);
}

double dgauss(double mean, double stdDev, int tid)
{
    return ThreadRand::dgauss(mean, stdDev, tid);
}

namespace poseSolver {
    
    std::pair<cv::Mat, cv::Mat> getInvHyp(const std::pair<cv::Mat, cv::Mat>& hyp)
    {
        cv::Mat_<double> hypR, trans = cv::Mat_<float>::eye(4, 4);
        cv::Rodrigues(hyp.first, hypR);

        hypR.copyTo(trans.rowRange(0,3).colRange(0,3));
        trans(0, 3) = hyp.second.at<double>(0, 0);
        trans(1, 3) = hyp.second.at<double>(0, 1);
        trans(2, 3) = hyp.second.at<double>(0, 2);

        trans = trans.inv();

        std::pair<cv::Mat, cv::Mat> invHyp;
        cv::Rodrigues(trans.rowRange(0,3).colRange(0,3), invHyp.first);
        invHyp.second = cv::Mat_<double>(1, 3);
        invHyp.second.at<double>(0, 0) = trans(0, 3);
        invHyp.second.at<double>(0, 1) = trans(1, 3);
        invHyp.second.at<double>(0, 2) = trans(2, 3);

        return invHyp;
    }

    double calcAngularDistance(const std::pair<cv::Mat, cv::Mat> & h1, const std::pair<cv::Mat, cv::Mat> & h2)
    {
        cv::Mat r1, r2;
        cv::Rodrigues(h1.first, r1);
        cv::Rodrigues(h2.first, r2);

        cv::Mat rotDiff= r2 * r1.t();
        double trace = cv::trace(rotDiff)[0];

        trace = std::min(3.0, std::max(-1.0, trace));
        return 180*acos((trace-1.0)/2.0)/CV_PI;
    }

    double maxLoss(const std::pair<cv::Mat, cv::Mat> & h1, const std::pair<cv::Mat, cv::Mat> & h2)
    {
        // measure loss of inverted poses (camera pose instead of scene pose)
        std::pair<cv::Mat, cv::Mat>  invH1 = getInvHyp(h1);
        std::pair<cv::Mat, cv::Mat>  invH2 = getInvHyp(h2);

        double rotErr = calcAngularDistance(invH1, invH2);
        double tErr = cv::norm(invH1.second - invH2.second);

        return std::max(rotErr, tErr * 100);
    }

    inline bool safeSolvePnP(
        const std::vector<cv::Point3f>& objPts,
        const std::vector<cv::Point2f>& imgPts,
        const cv::Mat& camMat,
        const cv::Mat& distCoeffs,
        cv::Mat& rot,
        cv::Mat& trans,
        bool extrinsicGuess,
        int methodFlag)
    {
        if(rot.type() == 0) rot = cv::Mat_<double>::zeros(1, 3);
        if(trans.type() == 0) trans= cv::Mat_<double>::zeros(1, 3);

        if(!cv::solvePnP(objPts, imgPts, camMat, distCoeffs, rot, trans, extrinsicGuess, methodFlag))
        {
            rot = cv::Mat_<double>::zeros(1, 3);
            trans = cv::Mat_<double>::zeros(1, 3);
            return false;
        }
        return true;
    }

    PnPRANSAC::PnPRANSAC () {
        this->camMat = cv::Mat_<float>::eye(3, 3);
    }
    
    PnPRANSAC::PnPRANSAC (float fx, float fy, float cx, float cy) {
        this->camMat = cv::Mat_<float>::eye(3, 3);
        this->camMat(0,0) = fx;
        this->camMat(1,1) = fy;
        this->camMat(0,2) = cx;
        this->camMat(1,2) = cy;
    }
    
    PnPRANSAC::~PnPRANSAC () {}
    
    void PnPRANSAC::camMatUpdate(float fx, float fy, float cx, float cy){
        this->camMat = cv::Mat_<float>::eye(3, 3);
        this->camMat(0,0) = fx;
        this->camMat(1,1) = fy;
        this->camMat(0,2) = cx;
        this->camMat(1,2) = cy;
    }




    double* PnPRANSAC::RANSAC_one2many(
        float* imgPts_,
        float* objPts_,
        int nPts,
        int nCps,
        int objHyps,
        int flag=0)
    {
        // hyper params.
        int inlierThreshold2D = 10; // 10 default.
        int refSteps = 100;
        // format data.
        std::vector<cv::Point2f> imgPts(nPts);
        std::vector<std::vector<cv::Point3f>> objPts(nPts);
        for (unsigned i = 0; i < objPts.size(); i++)
            objPts[i] = std::vector<cv::Point3f>(nCps);
        #pragma omp parallel for
        for(unsigned i=0; i<imgPts.size(); i++)
            imgPts[i] = cv::Point2f(imgPts_[i*2], imgPts_[i*2+1]); // default.
            //imgPts[i] = cv::Point2f(imgPts_[i*2]+4, imgPts_[i*2+1]+4); // shift.
        #pragma omp parallel for
        for(unsigned i=0; i<objPts.size(); i++)
            for (unsigned j = 0; j < nCps; j++)
                objPts[i][j] = cv::Point3f(objPts_[i*nCps*3+j*3], objPts_[i*nCps*3+j*3 + 1], objPts_[i*nCps*3+j*3 + 2]);
        // variables.
        std::vector<std::vector<cv::Point2f>> sampledImgPts(objHyps);
        std::vector<std::vector<cv::Point3f>> sampledObjPts(objHyps);
        std::vector<cv::Mat_<double>> rotHyp(objHyps);
        std::vector<cv::Mat_<double>> tHyp(objHyps);
        std::vector<float> scores(objHyps);
        std::vector<std::vector<float>> reproDiff(objHyps);
        std::vector<std::vector<int>> correspID(objHyps);
        // sample hypotheses.
        #pragma omp parallel for
        for(int h = 0; h < objHyps; h++)
        while(true)
        {
            std::vector<cv::Point2f> projections;
            std::vector<int> alreadyChosen(nPts,0);
            sampledImgPts[h].clear();
            sampledObjPts[h].clear();
            for(int j = 0; j < 4; j++)
            {
                int idx = irand(0, nPts);
                if(alreadyChosen[idx] > 0)
                {
                    j--;
                    continue;
                }
                int cpIdx = irand(0, nCps);
                alreadyChosen[idx] = 1;
                sampledImgPts[h].push_back(imgPts[idx]); // 2D coordinate.
                sampledObjPts[h].push_back(objPts[idx][cpIdx]); // 3D coordinate.
            }
            if(!safeSolvePnP(sampledObjPts[h], sampledImgPts[h], this->camMat, cv::Mat(), rotHyp[h], tHyp[h], false, CV_P3P)) continue;
            // check reconstruction, 4 sampled points should be reconstructed perfectly.
            cv::projectPoints(sampledObjPts[h], rotHyp[h], tHyp[h], this->camMat, cv::Mat(), projections);
            bool foundOutlier = false;
            for(unsigned j = 0; j < sampledImgPts[h].size(); j++)
            {
                if(cv::norm(sampledImgPts[h][j] - projections[j]) < inlierThreshold2D)
                    continue;
                foundOutlier = true;
                break;
            }
            if(foundOutlier)
                continue;
            else{
                // compute reprojection error and hypothesis score.
                std::vector<cv::Point2f> projections; 
                std::vector<cv::Point3f> cpPts(nPts*nCps);
                for (int i = 0; i < nPts; ++i)
                    for (int j = 0; j < nCps; ++j)
                        cpPts[i*nCps+j] = objPts[i][j];
                cv::projectPoints(cpPts, rotHyp[h], tHyp[h], this->camMat, cv::Mat(), projections);
                std::vector<float> diff(nPts);
                std::vector<int> cpID(nPts);
                float score = 0.;
                // #pragma omp parallel for
                for(unsigned pt = 0; pt < imgPts.size(); pt++)
                {
                    float err = 999.;
                    // min err among correspondences.
                    for (unsigned cp = 0; cp < nCps; cp++)
                    {
                        float residual = cv::norm(imgPts[pt] - projections[pt*nCps+cp]);
                        if (residual < err) 
                        {
                            err = residual;
                            cpID[pt] = cp;
                        }
                    }
                    if (flag!=0)
                    {
                        // mean err among correspondences.
                        err = 0.;
                        for (unsigned cp = 0; cp < nCps; cp++)
                        {
                            float residual = cv::norm(imgPts[pt] - projections[pt*nCps+cp]);
                            err += residual;
                        }
                        err /= nCps;                        
                    }
                    diff[pt] = err;
                    // compute the score.
                    score = score + (1. / (1. + std::exp(-(0.5*(err-inlierThreshold2D)))));
                }
                reproDiff[h] = diff;
                correspID[h] = cpID;
                scores[h] = score;
                break;
            }
        }
        // select one winning hypothesis.
        int hypIdx = std::min_element(scores.begin(),scores.end()) - scores.begin(); 

        // refine the hypothesis.
        double convergenceThresh = 0.01; // 0.01 by default.
        std::vector<float> localDiff = reproDiff[hypIdx];
        std::vector<int> localCpID = correspID[hypIdx];
        for(int rStep = 0; rStep < refSteps; rStep++)
        {
            // collect inliers.
            std::vector<cv::Point2f> localImgPts;
            std::vector<cv::Point3f> localObjPts;
            for(int pt = 0; pt < nPts; pt++)
            {
                if(localDiff[pt] < inlierThreshold2D)
                {
                    localImgPts.push_back(imgPts[pt]);
                    localObjPts.push_back(objPts[pt][localCpID[pt]]);
                }
            }
            if(localImgPts.size() < 4)
                break;
            // recalculate pose.
            cv::Mat_<double> rotNew = rotHyp[hypIdx].clone();
            cv::Mat_<double> tNew = tHyp[hypIdx].clone();
            if(!safeSolvePnP(localObjPts, localImgPts, this->camMat, cv::Mat(), rotNew, tNew, true, (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P))
                break; // abort if PnP fails.

            std::pair<cv::Mat, cv::Mat> hypNew;
            std::pair<cv::Mat, cv::Mat> hypOld;
            hypNew.first = rotNew;
            hypNew.second = tNew;
            hypOld.first = rotHyp[hypIdx];
            hypOld.second = tHyp[hypIdx];
            if(maxLoss(hypNew, hypOld) < convergenceThresh) // check convergence.
                break;

            rotHyp[hypIdx] = rotNew;
            tHyp[hypIdx] = tNew;
            // update correspondences.
            std::vector<cv::Point2f> projections;
            std::vector<cv::Point3f> cpPts(nPts*nCps);
            for (int i = 0; i < nPts; ++i)
                for (int j = 0; j < nCps; ++j)
                    cpPts[i*nCps+j] = objPts[i][j];
            cv::projectPoints(cpPts, rotHyp[hypIdx], tHyp[hypIdx], this->camMat, cv::Mat(), projections);
            std::vector<float> diff(nPts);
            std::vector<int> cpID(nPts);
            #pragma omp parallel for
            for(unsigned pt = 0; pt < imgPts.size(); pt++)
            {
                //if (rStep < 100)
                if (true)
                {
                    float err = 999.;
                    // min err among correspondences.
                    for (unsigned cp = 0; cp < nCps; cp++)
                    {
                        float residual = cv::norm(imgPts[pt] - projections[pt*nCps+cp]);
                        if (residual < err) 
                        {
                            err = residual;
                            cpID[pt] = cp;
                        }
                    }
                    if (flag!=0) // mean err among correspondences.
                    {
                        err = 0.;
                        for (unsigned cp = 0; cp < nCps; cp++)
                        {
                            float residual = cv::norm(imgPts[pt] - projections[pt*nCps+cp]);
                            err += residual;
                        }
                        err /= nCps;                        
                    }
                    diff[pt] = err;
                }
            }
            localDiff = diff;
            localCpID = cpID;
        }


        // // debug: save the inliers to file.
        // std::vector<cv::Point2f> inliers2D;
        // std::vector<cv::Point3f> inliers3D;
        // for(int pt = 0; pt < nPts; pt++)
        // {
        //     if(localDiff[pt] < inlierThreshold2D)
        //     {
        //         inliers2D.push_back(imgPts[pt]);
        //         inliers3D.push_back(objPts[pt][localCpID[pt]]);
        //     }
        // }
        // std::ofstream file2D, file3D;
        // file2D.open("_inliers2D.txt");
        // file3D.open("_inliers3D.txt");
        // for (int idx=0; idx<inliers2D.size(); idx++)
        // {
        //     file2D << inliers2D[idx].x << " " << inliers2D[idx].y << "\n";
        //     file3D << inliers3D[idx].x << " " << inliers3D[idx].y << " " << inliers3D[idx].z << "\n";
        // }
        // file2D.close();
        // file3D.close();


        static double  pose[6];
        for (int i = 0; i < 3; i++) 
            pose[i] = rotHyp[hypIdx](0,i);
        for (int i = 3; i < 6; i++) 
            pose[i] = tHyp[hypIdx](0,i-3);
        return pose;
    }
}

