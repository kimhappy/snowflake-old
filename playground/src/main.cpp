/*******************************************************************************
 *  Copyright 2023 Gaudio Lab, Inc.
 *  All rights reserved.
 *  http://gaudiolab.com
 ******************************************************************************/

#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <getopt.h>
#include <gwav_define.h>
#include <gwav_reader.h>
#include <gwav_writer.h>
#include <snowflake/model.hpp>
#include <lstm_6505.hpp>
#include <lstm_zendrive.hpp>
#include <gru_amp_gain.hpp>

auto main(
  int32_t  const argc,
  char   * const argv[]) noexcept -> int32_t {
  constexpr auto SR      = (int32_t)44100;
  constexpr auto OPTIONS = std::array {
    option { "input" , required_argument, 0, 'i' },
    option { "output", required_argument, 0, 'o' },
    option { "block" , required_argument, 0, 'b' },
    option { 0       , 0                , 0, 0   } };

  auto shrt = (int32_t)0;
  auto oidx = (int32_t)0;

  auto inPath  = std::string {};
  auto outPath = std::string {};
  auto spb     = (int32_t)0;

  while ((shrt = getopt_long(argc, argv, "i:o:b:", OPTIONS.data(), &oidx)) != -1) {
    switch (shrt) {
      case 'i':
        inPath  = optarg;
        break;

      case 'o':
        outPath = optarg;
        break;

      case 'b':
        spb     = std::atoi(optarg);
        break;
    }
  }

  auto reader = GWavIO::GWavReader {};
  auto writer = GWavIO::GWavWriter {};

  reader.createWavReader(inPath.data(), nullptr);
  writer.createWavWriter(outPath.data(), reader.getSampleRate(), reader.getNumChannels(), reader.getBitsPerSample());

  // auto model = snowflake::Model< float, snowflake::layer::LSTM, 40, 0 >((float const*)snowflake::model::LSTM_6505    );
  // auto model = snowflake::Model< float, snowflake::layer::LSTM, 40, 1 >((float const*)snowflake::model::LSTM_ZENDRIVE);
  auto model = snowflake::Model< float, snowflake::layer::GRU , 10, 0 >((float const*)snowflake::model::GRU_AMP_GAIN );

  auto ibuffer = std::vector< float >(spb);
  auto obuffer = std::vector< float >(spb);

  auto ts  = (uint64_t)0;
  auto cnt = (int64_t )0;

  for (; reader.readAudioSamples(ibuffer.data(), ibuffer.size()); ++cnt) {
    auto const begin     = std::chrono::high_resolution_clock::now();
    model.process(ibuffer.data(), obuffer.data(), spb);
    // model.process(ibuffer.data(), obuffer.data(), spb, 1.f);
    auto const end       = std::chrono::high_resolution_clock::now();
    auto const duration  = std::chrono::duration_cast< std::chrono::microseconds >(end - begin).count();
    ts                  += duration;
    writer.writeAudioSamples(obuffer.data(), obuffer.size());
  }

  auto const meandu = (double)ts / cnt;
  auto const rtf    = meandu / spb * SR / 1000000;

  std::cout << "Frm : " << cnt << '\n';
  std::cout << "Mean: " << meandu << "us (RTF: " << rtf << ")\n";

  reader.closeWavReader();
  writer.closeWavWriter();

  return 0;
}
